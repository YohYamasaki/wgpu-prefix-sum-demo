pub fn cpu_prefix_sum(data: &[u32]) -> Vec<u32> {
    if data.is_empty() {
        return vec![];
    }
    let n = data.len();
    let mut res = vec![0u32; n];
    res[0] = data[0];
    for i in 1..n {
        res[i] = res[i - 1] + data[i];
    }
    res
}
