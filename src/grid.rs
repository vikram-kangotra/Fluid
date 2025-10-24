#[derive(Clone)]
pub struct Grid {
    pub n: usize,
    data: Vec<f32>, // length = n*n
}

impl Grid {
    /// Create a new grid of size n x n filled with `val`.
    pub fn new(n: usize, val: f32) -> Self {
        Self { n, data: vec![val; n * n] }
    }

    #[inline(always)]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        x + y * self.n
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[self.idx(x, y)]
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        let i = self.idx(x, y);
        self.data[i] = v;
    }
}
