use std::sync::mpsc::channel;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub fn cpu_prefix_sum(data: &[u32]) -> Vec<u32> {
    let n = data.len();
    let mut res = vec![0u32; n];
    res[0] = data[0];
    for i in 1..n {
        res[i] = res[i - 1] + data[i];
    }
    res
}

pub async fn init_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No adapter found");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: Default::default(),
        })
        .await
        .expect("Failed to create device");
    (device, queue)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    step: u32,
    _pad: [u32; 3],
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    data0: wgpu::Buffer,
    data1: wgpu::Buffer,
    uniform: wgpu::Buffer,
    readback: wgpu::Buffer,
    n: usize,
    max_steps: u32,
}

impl GpuContext {
    pub async fn new(n: usize) -> anyhow::Result<Self> {
        let (device, queue) = init_wgpu().await;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("hillis-steele-scan.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefix-sum"),
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let byte_len = (n * size_of::<u32>()) as u64;

        let data0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("data0"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let data1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("data1"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg0"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg1"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        let max_steps = n.next_power_of_two().ilog2(); // ceil(log2(N))

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_0,
            bind_group_1,
            data0,
            data1,
            uniform,
            readback,
            n,
            max_steps,
        })
    }

    pub fn upload_data(&self, input: &[u32]) {
        self.queue
            .write_buffer(&self.data0, 0, bytemuck::cast_slice(input));
    }

    pub fn read_computed_data(&self) -> anyhow::Result<Vec<u32>> {
        // Copy the result buffer to the temp buffer to bring the data to the CPU land
        let result_buf = if self.max_steps == 0 {
            &self.data0
        } else if (self.max_steps - 1) % 2 == 0 {
            &self.data1
        } else {
            &self.data0
        };
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&result_buf, 0, &self.readback, 0, self.data0.size());

        self.queue.submit([encoder.finish()]);

        let slice = self.readback.slice(..);
        // The mapping process is async, so we'll need to create a channel to get
        // the success flag for our mapping
        let (tx, rx) = channel();

        // We send the success or failure of our mapping via a callback
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // The callback we submitted to map async will only get called after the
        // device is polled or the queue submitted
        self.device.poll(wgpu::PollType::wait_indefinitely())?;

        // We check if the mapping was successful here
        rx.recv()??;

        // We then get the bytes that were stored in the buffer
        let bytes = slice.get_mapped_range();
        let out_u32: &[u32] = bytemuck::cast_slice(&bytes);
        Ok(out_u32.to_vec())
    }

    pub fn run_prefix_scan(&self) {
        let num_dispatches = self.n.div_ceil(64) as u32;

        for i in 0..self.max_steps {
            let step = 1u32 << i;
            let uni = Uniforms { step, _pad: [0; 3] };
            self.queue
                .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uni));

            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(
                    0,
                    if i % 2 == 0 {
                        &self.bind_group_0
                    } else {
                        &self.bind_group_1
                    },
                    &[],
                );
                pass.dispatch_workgroups(num_dispatches, 1, 1);
            }
            self.queue.submit([encoder.finish()]);
        }
    }
}

pub async fn gpu_prefix_sum(data: &[u32]) -> anyhow::Result<Vec<u32>> {
    let (device, queue) = init_wgpu().await;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("hillis-steele-scan.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    // Buffers for ping-pong
    let data_buffer_0 = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("data buffer 0"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
    });
    let data_buffer_1 = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("data buffer 1"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::STORAGE,
    });
    // Uniform buffer to pass current step
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("step buffer"),
        size: 16, // u32
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Temporary buffer to write out the result
    let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("temp buffer"),
        size: data_buffer_0.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Bind groups to do ping-pong data transfer
    let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind group 2"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: data_buffer_0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data_buffer_1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });
    let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind group 1"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: data_buffer_1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data_buffer_0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });
    let max_steps = data.len().next_power_of_two().ilog2(); // ceil(log2(N))
    {
        // Required number of dispatches to invocate computation more than the data element number
        let num_dispatches = data.len().div_ceil(64) as u32;
        for i in 0..max_steps {
            let step: u32 = 1 << i;
            // Update uniform buffer to pass the current step
            let uni = Uniforms { step, _pad: [0; 3] };
            queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uni));
            // Create a compute pass and submit it. Ideally, it is better to use one queue.
            // submit for efficiency, but this requires a dynamic offset or a 2N array with manual offsetting,
            // which requires more boilerplate and may be slower than submitting ceil(logN) times.
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(
                    0,
                    if i % 2 == 0 {
                        &bind_group_0
                    } else {
                        &bind_group_1
                    },
                    &[],
                );
                pass.dispatch_workgroups(num_dispatches, 1, 1);
            }
            queue.submit([encoder.finish()]);
        }
    }

    // Copy the result buffer to the temp buffer to bring the data to the CPU land
    let result_buf = if max_steps == 0 {
        &data_buffer_0
    } else if (max_steps - 1) % 2 == 0 {
        &data_buffer_1
    } else {
        &data_buffer_0
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&result_buf, 0, &temp_buffer, 0, data_buffer_0.size());

    queue.submit([encoder.finish()]);

    let slice = temp_buffer.slice(..);
    // The mapping process is async, so we'll need to create a channel to get
    // the success flag for our mapping
    let (tx, rx) = channel();

    // We send the success or failure of our mapping via a callback
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    // The callback we submitted to map async will only get called after the
    // device is polled or the queue submitted
    device.poll(wgpu::PollType::wait_indefinitely())?;

    // We check if the mapping was successful here
    rx.recv()??;

    // We then get the bytes that were stored in the buffer
    let bytes = slice.get_mapped_range();
    let out_u32: &[u32] = bytemuck::cast_slice(&bytes);
    Ok(out_u32.to_vec())
}
