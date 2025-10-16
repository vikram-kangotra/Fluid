pub struct FluidCube {
    size: usize,

    dt: f32,
    diff: f32,
    visc: f32,

    s: Vec<Vec<f32>>,
    pub density: Vec<Vec<f32>>,

    pub vx: Vec<Vec<f32>>,
    pub vy: Vec<Vec<f32>>,

    ux: Vec<Vec<f32>>,
    uy: Vec<Vec<f32>>,
}

impl FluidCube {

    pub fn new(size: usize, diffusion: f32, viscosity: f32, dt: f32) -> Self {
        Self {
            size,
            dt,
            diff: diffusion,
            visc: viscosity,

            s: vec![vec![0.0; size]; size],
            density: vec![vec![0.0; size]; size],

            vx: vec![vec![0.0; size]; size],
            vy: vec![vec![0.0; size]; size],

            ux: vec![vec![0.0; size]; size],
            uy: vec![vec![0.0; size]; size],
        }
    }

    pub fn add_density(&mut self, x: usize, y: usize, amount: f32) {
        self.density[x][y] += amount;
    }

    pub fn add_velocity(&mut self, x: usize, y: usize, amountx: f32, amounty: f32)
    {
        self.vx[x][y] += amountx;
        self.vy[x][y] += amounty;
    }

    pub fn step(&mut self, iter: usize) {
        Self::diffuse(1, &mut self.ux, &self.vx, self.visc, self.dt, iter, self.size);
        Self::diffuse(2, &mut self.uy, &self.vy, self.visc, self.dt, iter, self.size);

        Self::project(&mut self.ux, &mut self.uy,
            &mut self.vx, &mut self.vy, iter, self.size);

        Self::advect(1, &mut self.vx, &self.ux, &self.ux, &self.uy, self.dt, self.size);
        Self::advect(2, &mut self.vy, &self.uy, &self.ux, &self.uy, self.dt, self.size);

        Self::project(&mut self.vx, &mut self.vy, &mut self.ux, &mut self.uy, iter, self.size);

        Self::diffuse(0, &mut self.s, &self.density, self.diff, self.dt, iter, self.size);
        Self::advect(0, &mut self.density, &self.s, &self.vx, &self.vy, self.dt, self.size);
    }

    fn set_bound(b: usize, x: &mut Vec<Vec<f32>>, n: usize) {

        // XZ faces at y=0 and y=N-1
        for i in 1..n-1 {
            x[i][0] = if b == 2 { -x[i][1] } else { x[i][1] };
            x[i][n-1] = if b == 2 { -x[i][n-2] } else { x[i][n-2] };
        }

        // YZ faces at x=0 and x=N-1
        for j in 1..n-1 {
            x[0][j] = if b == 1 { -x[1][j] } else { x[1][j] };
            x[n-1][j] = if b == 1 { -x[n-2][j] } else { x[n-2][j] };
        }

        x[0][0]       = 0.5 * (x[1][0] + x[0][1]);
        x[0][n-1]     = 0.5 * (x[1][n-1] + x[0][n-2]);
        x[n-1][0]     = 0.5 * (x[n-2][0] + x[n-1][1]);
        x[n-1][n-1]   = 0.5 * (x[n-2][n-1] + x[n-1][n-2]);
    }

    fn lin_solve(b: usize, x: &mut Vec<Vec<f32>>, x0: &Vec<Vec<f32>>,
                a: f32, c: f32, iter: usize, n: usize)
    {
        let c_recip = 1.0 / c;
        for _ in 0..iter {
            for j in 1..n-1 {
                for i in 1..n-1 {
                    x[i][j] = (x0[i][j] + 
                        a * (
                            x[i + 1][j] +
                            x[i - 1][j] +
                            x[i][j + 1] +
                            x[i][j - 1]
                        )) * c_recip;
                }
            }
            Self::set_bound(b, x, n);
        }
    }

    fn diffuse(b: usize, x: &mut Vec<Vec<f32>>, x0: &Vec<Vec<f32>>, diff: f32, dt: f32, iter: usize, n: usize)
    {
        let a = dt * diff * ((n - 2) * (n - 2)) as f32;
        Self::lin_solve(b, x, x0, a, 1f32 + a * 6f32, iter, n);
    }

    fn project(velx: &mut Vec<Vec<f32>>, vely: &mut Vec<Vec<f32>>,
        p: &mut Vec<Vec<f32>>, div: &mut Vec<Vec<f32>>, iter: usize, n: usize)
    {
        let n_recip = 1f32 / n as f32;
        for j in 1..n-1 {
            for i in 1..n-1 {
                div[i][j] = -0.5f32 * (
                    velx[i + 1][j] - velx[i - 1][j] +
                    vely[i][j + 1] - vely[i][j - 1]
                ) * n_recip;
            }
        }

        Self::set_bound(0, div, n);
        Self::set_bound(0, p, n);
        Self::lin_solve(0, p, div, 1f32, 6f32, iter, n);

        for j in 1..n-1 {
            for i in 1..n-1 {
                velx[i][j] -= 0.5f32 * (p[i+1][j] - p[i-1][j]) * n as f32;
                vely[i][j] -= 0.5f32 * (p[i][j+1] - p[i][j-1]) * n as f32;
            }
        }

        Self::set_bound(1, velx, n);
        Self::set_bound(2, vely, n);
    }

    fn advect(b: usize, d: &mut Vec<Vec<f32>>, d0: &Vec<Vec<f32>>,
        velx: &Vec<Vec<f32>>, vely: &Vec<Vec<f32>>,
        dt: f32, n: usize)
    {
        let dtx = dt * (n - 2) as f32;
        let dty = dt * (n - 2) as f32;

        for j in 1..n-1 {
            for i in 1..n-1 {
                let tmp1 = dtx * velx[i][j];
                let tmp2 = dty * vely[i][j];

                let mut x = i as f32 - tmp1;
                let mut y = j as f32 - tmp2;

                x = x.clamp(0.5f32, (n - 1) as f32 + 0.5f32);
                let i0 = x.floor();
                let i1 = i0 + 1f32;

                y = y.clamp(0.5f32, (n - 1) as f32 + 0.5f32);
                let j0 = y.floor();
                let j1 = j0 + 1.0f32;

                let s1 = x - i0;
                let s0 = 1.0f32 - s1;
                let t1 = y - j0;
                let t0 = 1.0f32 - t1;

                let i0i = i0 as usize;
                let i1i = (i1 as usize).min(n - 1);
                let j0i = j0 as usize;
                let j1i = (j1 as usize).min(n - 1);
 
                d[i][j] =
                    s0 * (
                        t0 * d0[i0i][j0i] +
                        t1 * d0[i0i][j1i]
                    ) +
                    s1 * (
                        t0 * d0[i1i][j0i] +
                        t1 * d0[i1i][j1i]
                    );
            }
        }
        Self::set_bound(b, d, n);
    }
}


