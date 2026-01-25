use crate::utils::{align_up, init_wgpu};
use std::num::NonZeroU64;
use std::sync::mpsc::channel;
use wgpu::util::DeviceExt;

fn split_dispatch_3d(workgroups_needed: u32, max_dim: u32) -> [u32; 3] {
    let x = workgroups_needed.min(max_dim);
    let remaining_after_x = (workgroups_needed + x - 1) / x;
    let y = remaining_after_x.min(max_dim);

    let xy = (x as u64) * (y as u64);
    let z = ((workgroups_needed as u64) + xy - 1) / xy;
    assert!(z <= max_dim as u64, "dispatch exceeds max_dim^3");

    [x, y, z as u32]
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    step: u32, // this has to be a power of 2
    _pad: [u32; 3],
}
pub struct BlellochGpuContext {
    device: wgpu::Device,
    pub queue: wgpu::Queue,
    up_sweep_pipeline: wgpu::ComputePipeline,
    last_zero_pipeline: wgpu::ComputePipeline,
    down_sweep_pipeline: wgpu::ComputePipeline,
    up_sweep_bind_group: wgpu::BindGroup,
    last_zero_bind_group: wgpu::BindGroup,
    down_sweep_bind_group: wgpu::BindGroup,
    data: wgpu::Buffer,
    uniform: wgpu::Buffer,
    readback: wgpu::Buffer,
    n: usize,
    max_steps: u32,
    uniform_stride: u32,
}

impl BlellochGpuContext {
    pub async fn new(n: usize) -> anyhow::Result<Self> {
        assert!(
            n.is_power_of_two(),
            "Number of elements of data has to be a power of 2."
        );

        let (device, queue) = init_wgpu().await;

        let up_sweep_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("up-sweep shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blelloch_scan_up_sweep.wgsl").into()),
        });

        let last_zero_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("set-last_zero shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("set_last_zero.wgsl").into()),
        });

        let down_sweep_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("down-sweep shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blelloch_scan_down_sweep.wgsl").into()),
        });

        let sweep_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prefix-sum bgl"),
                entries: &[
                    // data: storage read & write
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // uni: uniform data for steps
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            // We will store all the steps into one uniform, that requires to have dynamic offset
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(size_of::<Uniform>() as u64),
                        },
                        count: None,
                    },
                ],
            });

        let sweep_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("prefix-sum pipeline layout"),
                bind_group_layouts: &[&sweep_bind_group_layout],
                immediate_size: 0,
            });

        let up_sweep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("up-sweep pipeline"),
            layout: Some(&sweep_pipeline_layout),
            module: &up_sweep_shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let last_zero_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("last zero pipeline"),
            layout: None,
            module: &last_zero_shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let down_sweep_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("down-sweep pipeline"),
                layout: Some(&sweep_pipeline_layout),
                module: &down_sweep_shader,
                entry_point: None,
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let byte_len = (n * size_of::<u32>()) as u64;

        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("data"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let max_steps = n.next_power_of_two().ilog2();

        // Calculate stride between uniforms in the aggregate buffer
        let align = device.limits().min_uniform_buffer_offset_alignment as usize;
        let uni_size = size_of::<Uniform>();
        let stride = align_up(uni_size, align);
        let uniform_stride = stride as u32;

        // Create a byte array of the uniforms with the stride
        let mut blob = vec![0u8; stride * (max_steps as usize)];
        for i in 0..max_steps {
            let u = Uniform {
                step: 2u32 << i,
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

        let up_sweep_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("up-sweep bind group"),
            layout: &up_sweep_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uni_binding.clone(),
                },
            ],
        });

        let last_zero_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg0"),
            layout: &last_zero_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: data.as_entire_binding(),
            }],
        });

        let down_sweep_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("down-sweep bind group"),
            layout: &up_sweep_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uni_binding,
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            up_sweep_pipeline,
            last_zero_pipeline,
            down_sweep_pipeline,
            up_sweep_bind_group,
            last_zero_bind_group,
            down_sweep_bind_group,
            data,
            uniform,
            readback,
            n,
            max_steps,
            uniform_stride,
        })
    }

    pub fn upload_data(&self, input: &[u32]) {
        self.queue
            .write_buffer(&self.data, 0, bytemuck::cast_slice(input));
    }

    pub fn read_computed_data(&self) -> anyhow::Result<Vec<u32>> {
        // Copy the result buffer to the temp buffer to bring the data to the CPU land
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.data, 0, &self.readback, 0, self.data.size());

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

    pub fn encode_up_sweep(&self, encoder: &mut wgpu::CommandEncoder) {
        const WG_SIZE: u32 = 64;
        let max_dim = self.device.limits().max_compute_workgroups_per_dimension;

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.up_sweep_pipeline);
        for i in 0..self.max_steps {
            let step = 2u32 << i; // same as uniform
            let active = self.n as u32 / step;
            let workgroups_needed = active.div_ceil(WG_SIZE).max(1);
            let [x, y, z] = split_dispatch_3d(workgroups_needed, max_dim);

            let offset_bytes = i * self.uniform_stride;
            pass.set_bind_group(0, &self.up_sweep_bind_group, &[offset_bytes]);
            pass.dispatch_workgroups(x, y, z);
        }
    }

    pub fn encode_set_last_zero(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.last_zero_pipeline);
        pass.set_bind_group(0, &self.last_zero_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub fn encode_down_sweep(&self, encoder: &mut wgpu::CommandEncoder) {
        const WG_SIZE: u32 = 64;
        let max_dim = self.device.limits().max_compute_workgroups_per_dimension;

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.down_sweep_pipeline);

        for i in (0..self.max_steps).rev() {
            let step = 2u32 << i; // same as uniform
            let active = self.n as u32 / step;
            let workgroups_needed = active.div_ceil(WG_SIZE).max(1);
            let [x, y, z] = split_dispatch_3d(workgroups_needed, max_dim);

            let offset_bytes = i * self.uniform_stride;
            pass.set_bind_group(0, &self.down_sweep_bind_group, &[offset_bytes]);
            pass.dispatch_workgroups(x, y, z);
        }
    }

    pub fn get_command_encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&Default::default())
    }

    pub fn run_prefix_sum(&self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_up_sweep(&mut encoder);
        self.encode_set_last_zero(&mut encoder);
        self.encode_down_sweep(&mut encoder);
        self.submit(encoder);
    }
    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
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
