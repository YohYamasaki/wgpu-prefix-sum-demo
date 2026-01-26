use crate::utils::init_wgpu;
use std::sync::mpsc::channel;
fn split_dispatch_3d(workgroups_needed: u32, max_dim: u32) -> [u32; 3] {
    let x = workgroups_needed.min(max_dim);
    let remaining_after_x = (workgroups_needed + x - 1) / x;
    let y = remaining_after_x.min(max_dim);

    let xy = (x as u64) * (y as u64);
    let z = ((workgroups_needed as u64) + xy - 1) / xy;
    assert!(z <= max_dim as u64, "dispatch exceeds max_dim^3");

    [x, y, z as u32]
}

pub struct BlockBlellochGpuContext {
    device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipeline_write_sum: wgpu::ComputePipeline,
    pipeline_no_sum: wgpu::ComputePipeline,
    pipeline_add_carry: wgpu::ComputePipeline,
    bind_groups_write_sum: Vec<wgpu::BindGroup>,
    bind_group_no_sum: wgpu::BindGroup,
    bind_groups_add_carry: Vec<wgpu::BindGroup>,
    data_buffers: Vec<wgpu::Buffer>,
    elms_per_level: Vec<u32>,
    readback: wgpu::Buffer,
    n: usize,
}

impl BlockBlellochGpuContext {
    pub async fn new(n: usize) -> anyhow::Result<Self> {
        assert!(
            n.is_power_of_two(),
            "Number of elements of data has to be a power of 2."
        );

        let (device, queue) = init_wgpu().await;

        let block_scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("block-scan shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blelloch_block_scan.wgsl").into()),
        });

        let add_carry_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("add-carry shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blelloch_add_carry.wgsl").into()),
        });

        let pipeline_write_sum = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("block_scan_write_sum pipeline"),
            layout: None,
            module: &block_scan_shader,
            entry_point: Some("block_scan_write_sum"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let pipeline_no_sum = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("block_scan_no_sum pipeline"),
            layout: None,
            module: &block_scan_shader,
            entry_point: Some("block_scan_no_sum"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let pipeline_add_carry = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("add_carry pipeline"),
            layout: None,
            module: &add_carry_shader,
            entry_point: Some("add_carry"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        // Build all required buffers + block scan bind groups for each level
        const TILE_SIZE: usize = 64;
        let mut data_buffers: Vec<wgpu::Buffer> = vec![];
        let mut bind_groups_write_sum: Vec<wgpu::BindGroup> = vec![];
        let mut elms_per_level: Vec<u32> = vec![];
        // For original data
        data_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block-sum"),
            size: (n * size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        // Create buffers for blocks
        let mut level_elms = n;
        let mut i = 1;
        while level_elms > TILE_SIZE {
            elms_per_level.push(level_elms as u32);
            let num_blocks = level_elms.div_ceil(TILE_SIZE).max(1);
            let sum_bytes = (num_blocks * size_of::<u32>()) as u64;
            data_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("block-sum"),
                size: sum_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // bind group: (prev_level -> this_level)
            let src = &data_buffers[i - 1];
            let dst = &data_buffers[i];
            bind_groups_write_sum.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("block-scan bind group"),
                layout: &pipeline_write_sum.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                ],
            }));

            level_elms = num_blocks;
            i += 1;
        }
        // The last buffer's elements number is for `block_scan_no_sum`
        elms_per_level.push(level_elms as u32);

        let last_buffer = &data_buffers[data_buffers.len() - 1];
        let bind_group_no_sum = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block-scan bind group"),
            layout: &pipeline_no_sum.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: last_buffer.as_entire_binding(),
            }],
        });

        // Build Add-carry bind groups
        let mut bind_groups_add_carry: Vec<wgpu::BindGroup> = vec![];
        for i in (1..data_buffers.len()).rev() {
            bind_groups_add_carry.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("add-carry bind group"),
                layout: &pipeline_add_carry.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buffers[i - 1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: data_buffers[i].as_entire_binding(),
                    },
                ],
            }));
        }

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (n * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline_write_sum,
            pipeline_no_sum,
            pipeline_add_carry,
            bind_groups_write_sum,
            bind_group_no_sum,
            bind_groups_add_carry,
            data_buffers,
            elms_per_level,
            readback,
            n,
        })
    }

    pub fn upload_data(&self, input: &[u32]) {
        self.queue
            .write_buffer(&self.data_buffers[0], 0, bytemuck::cast_slice(input));
    }

    pub fn read_computed_data(&self) -> anyhow::Result<Vec<u32>> {
        // Copy the result buffer to the temp buffer to bring the data to the CPU land
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.data_buffers[0],
            0,
            &self.readback,
            0,
            self.data_buffers[0].size(),
        );

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

    pub fn encode_scan(&self, encoder: &mut wgpu::CommandEncoder) {
        const WG_SIZE: u32 = 64;
        let max_dim = self.device.limits().max_compute_workgroups_per_dimension;

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline_write_sum);

        // apply the scan for block sums recursively until the size of the block sums array becomes smaller than one block size
        self.bind_groups_write_sum
            .iter()
            .enumerate()
            .for_each(|(i, bind_group)| {
                let workgroups_needed = self.elms_per_level[i].div_ceil(WG_SIZE).max(1);
                pass.set_bind_group(0, bind_group, &[]);
                let [x, y, z] = split_dispatch_3d(workgroups_needed, max_dim);
                pass.dispatch_workgroups(x, y, z);
            });

        // The last sums also requires scan but no need to write the new block sums since it is already fitting in one block
        let last_idx = self.elms_per_level.len() - 1;
        let workgroups_needed = self.elms_per_level[last_idx].div_ceil(WG_SIZE).max(1);
        pass.set_pipeline(&self.pipeline_no_sum);
        pass.set_bind_group(0, &self.bind_group_no_sum, &[]);
        let [x, y, z] = split_dispatch_3d(workgroups_needed, max_dim);
        pass.dispatch_workgroups(x, y, z);

        // add carry to the previous data
        pass.set_pipeline(&self.pipeline_add_carry);
        for level in (1..self.data_buffers.len()).rev() {
            let bind_group = &self.bind_groups_add_carry[self.data_buffers.len() - 1 - level];
            let block_len = self.elms_per_level[level - 1];
            let workgroups_needed = block_len.div_ceil(WG_SIZE).max(1);

            pass.set_bind_group(0, bind_group, &[]);
            let [x, y, z] = split_dispatch_3d(workgroups_needed, max_dim);
            pass.dispatch_workgroups(x, y, z);
        }
    }

    pub fn get_command_encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&Default::default())
    }

    pub fn run_prefix_sum(&self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_scan(&mut encoder);
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
