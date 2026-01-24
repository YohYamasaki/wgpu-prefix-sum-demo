use std::num::NonZeroU64;
use std::sync::mpsc::channel;
use wgpu::util::DeviceExt;

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) / a * a
}

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

    let mut limits = wgpu::Limits::default();
    limits.max_buffer_size = adapter.limits().max_buffer_size;
    limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: Default::default(),
            required_limits: limits,
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: Default::default(),
        })
        .await
        .expect("Failed to create device");
    (device, queue)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    step: u32,
    _pad: [u32; 3],
}

pub struct GpuContext {
    device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    data0: wgpu::Buffer,
    data1: wgpu::Buffer,
    readback: wgpu::Buffer,
    n: usize,
    max_steps: u32,
    uniform_stride: u32,
}

impl GpuContext {
    pub async fn new(n: usize) -> anyhow::Result<Self> {
        let (device, queue) = init_wgpu().await;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("hillis-steele-scan.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix-sum bgl"),
            entries: &[
                // src: storage read
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // dst: storage read_write
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // uni: uniform (dynamic offset!)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // We will store all the steps into one uniform, that requires to have dynamic offset
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(size_of::<Uniforms>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefix-sum pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefix-sum"),
            layout: Some(&pipeline_layout),
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

        let max_steps = n.next_power_of_two().ilog2();

        // Calculate stride between uniforms in the aggregate buffer
        let align = device.limits().min_uniform_buffer_offset_alignment as usize;
        let uni_size = size_of::<Uniforms>();
        let stride = align_up(uni_size, align);
        let uniform_stride = stride as u32;

        // Create a byte array of the uniforms with the stride
        let mut blob = vec![0u8; stride * (max_steps as usize)];
        for i in 0..max_steps {
            let u = Uniforms {
                step: 1u32 << i,
                _pad: [0; 3],
            };
            let bytes = bytemuck::bytes_of(&u);
            let offset = (i as usize) * stride;
            blob[offset..offset + bytes.len()].copy_from_slice(bytes);
        }

        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform"),
            contents: &blob,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let uni_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &uniform,
            offset: 0,
            size: NonZeroU64::new(uni_size as u64),
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
                    resource: uni_binding.clone(),
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
                    resource: uni_binding,
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_0,
            bind_group_1,
            data0,
            data1,
            readback,
            n,
            max_steps,
            uniform_stride,
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
        let v = out_u32.to_vec();

        drop(bytes);
        self.readback.unmap();

        Ok(v)
    }

    pub fn run_prefix_scan(&self) {
        const WG_SIZE: u32 = 64;

        let workgroups_needed = self.n.div_ceil(WG_SIZE as usize) as u32;

        let max_dim = self.device.limits().max_compute_workgroups_per_dimension;
        let x = workgroups_needed.min(max_dim);
        let y = (workgroups_needed + x - 1) / x;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            for i in 0..self.max_steps {
                let offset_bytes = i * self.uniform_stride;
                let bg = if i % 2 == 0 {
                    &self.bind_group_0
                } else {
                    &self.bind_group_1
                };
                pass.set_bind_group(0, bg, &[offset_bytes]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        self.queue.submit([encoder.finish()]);
    }

    pub fn wait_for_previous_submit(&self) -> anyhow::Result<()> {
        let (tx, rx) = channel();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        let _ = rx.recv();
        Ok(())
    }

    pub fn wait_idle(&self) -> anyhow::Result<()> {
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }
}
