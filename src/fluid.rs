//! Fluid simulation (Stable Fluids) using flat Grid.
//! Implements diffusion, advection (semi-Lagrangian), and projection (incompressibility).
//!
//! Math reminders (2D):
//!   Diffusion PDE: ∂u/∂t = ν ∇² u -> implicit solve (I - a Δ) x = x0
//!   Advection: trace backwards x_prev = x - dt * v(x), then bilinear sample d0(x_prev)
//!   Projection: solve ∇² p = div, then u -= 0.5 * grad(p) * n

use crate::grid::Grid;

pub struct Fluid {
    pub dt: f32,
    pub diff: f32,
    pub visc: f32,

    pub s: Grid,
    pub density: Grid,

    pub vx: Grid,
    pub vy: Grid,

    pub ux: Grid,
    pub uy: Grid,

    obstacle: Grid,
}

impl Fluid {
    pub fn new(n: usize, diff: f32, visc: f32, dt: f32) -> Self {
        let mut fluid = Self {
            dt,
            diff,
            visc,

            s: Grid::new(n, 0.0),
            density: Grid::new(n, 0.0),

            vx: Grid::new(n, 0.0),
            vy: Grid::new(n, 0.0),

            ux: Grid::new(n, 0.0),
            uy: Grid::new(n, 0.0),

            obstacle: Grid::new(n, 0.0),
        };

        fluid.add_circular_obstacle();
        fluid
    }

    fn add_circular_obstacle(&mut self) {
        let cx = self.obstacle.n as f32 / 2.0;
        let cy = self.obstacle.n as f32 / 2.0;
        let radius = 5.0;
        let r2 = radius * radius;

        for y in 0..self.obstacle.n {
            for x in 0..self.obstacle.n {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx*dx + dy*dy <= r2 {
                    self.obstacle.set(x, y, 1.0);
                }
            }
        }
    }

    /// Add density at grid cell (x,y)
    pub fn add_density(&mut self, x: usize, y: usize, amount: f32) {
        if self.obstacle.get(x, y) == 0.0 {
            let val = self.density.get(x, y);
            self.density.set(x, y, val + amount);
        }
    }

    /// Add velocity component at grid cell (x,y)
    pub fn add_velocity(&mut self, x: usize, y: usize, amountx: f32, amounty: f32) {
        if self.obstacle.get(x, y) == 0.0 {
            let vx = self.vx.get(x, y);
            self.vx.set(x, y, vx + amountx);
            let vy = self.vy.get(x, y);
            self.vy.set(x, y, vy + amounty);
        }
    }

    /// Advance simulation by one timestep (iter = Gauss-Seidel iterations for solvers)
    pub fn step(&mut self, iter: usize) {
        diffuse(BoundType::Vx, &mut self.ux, &self.vx, self.visc, self.dt, iter);
        diffuse(BoundType::Vy, &mut self.uy, &self.vy, self.visc, self.dt, iter);
        project(&mut self.ux, &mut self.uy, &mut self.vx, &mut self.vy, iter);

        advect(BoundType::Vx, &mut self.vx, &self.ux, &self.ux, &self.uy, self.dt);
        advect(BoundType::Vy, &mut self.vy, &self.uy, &self.ux, &self.uy, self.dt);

        project(&mut self.vx, &mut self.vy, &mut self.ux, &mut self.uy, iter);

        diffuse(BoundType::Scalar, &mut self.s, &self.density, self.diff, self.dt, iter);
        advect(BoundType::Scalar, &mut self.density, &self.s, &self.vx, &self.vy, self.dt);

        self.apply_obstacle();
    }

    fn apply_obstacle(&mut self) {
        let n = self.obstacle.n;
        for y in 0..n {
            for x in 0..n {
                if self.obstacle.get(x, y) == 1.0 {
                    self.vx.set(x, y, 0.0);
                    self.vy.set(x, y, 0.0);
                    self.density.set(x, y, 0.0);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BoundType {
    Scalar,
    Vx,
    Vy,
}

/// Apply boundary conditions on `grid` for boundary type `b`.
fn set_bound(b: BoundType, grid: &mut Grid) {
    let n = grid.n;
    // top/bottom (y = 0, y = n-1)
    for i in 1..n - 1 {
        let v1 = grid.get(i, 1);
        let vn = grid.get(i, n - 2);
        grid.set(i, 0, if matches!(b, BoundType::Vy) { -v1 } else { v1 });
        grid.set(i, n - 1, if matches!(b, BoundType::Vy) { -vn } else { vn });
    }

    // left/right (x = 0, x = n-1)
    for j in 1..n - 1 {
        let v1 = grid.get(1, j);
        let vn = grid.get(n - 2, j);
        grid.set(0, j, if matches!(b, BoundType::Vx) { -v1 } else { v1 });
        grid.set(n - 1, j, if matches!(b, BoundType::Vx) { -vn } else { vn });
    }

    // corners (average)
    grid.set(0, 0, 0.5 * (grid.get(1, 0) + grid.get(0, 1)));
    grid.set(0, n - 1, 0.5 * (grid.get(1, n - 1) + grid.get(0, n - 2)));
    grid.set(n - 1, 0, 0.5 * (grid.get(n - 2, 0) + grid.get(n - 1, 1)));
    grid.set(n - 1, n - 1, 0.5 * (grid.get(n - 2, n - 1) + grid.get(n - 1, n - 2)));
}

/// Gauss–Seidel linear solver for (I - a Δ) x = x0
fn lin_solve(b: BoundType, x: &mut Grid, x0: &Grid, a: f32, c: f32, iter: usize) {
    let n = x.n;
    let c_recip = 1.0 / c;
    for _ in 0..iter {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let sum = x.get(i + 1, j)
                    + x.get(i - 1, j)
                    + x.get(i, j + 1)
                    + x.get(i, j - 1);
                let new = (x0.get(i, j) + a * sum) * c_recip;
                x.set(i, j, new);
            }
        }
        set_bound(b, x);
    }
}

/// Diffuse wrapper. a = dt * diff * (N-2)^2. In 2D use c = 1 + 4a.
fn diffuse(b: BoundType, x: &mut Grid, x0: &Grid, diff: f32, dt: f32, iter: usize) {
    let n = x.n;
    let a = dt * diff * ((n - 2) * (n - 2)) as f32;
    lin_solve(b, x, x0, a, 1.0 + 6.0 * a, iter); // delibrately set to 6.0 so that the smoke
                                                    // vanishes after some time
}

/// Semi-Lagrangian advection with bilinear interpolation.
/// d: output, d0: source field, velx/vely: velocities.
fn advect(b: BoundType, d: &mut Grid, d0: &Grid, velx: &Grid, vely: &Grid, dt: f32) {
    let n = d.n;
    let dt0 = dt * (n as f32 - 2.0);

    for j in 1..n - 1 {
        for i in 1..n - 1 {
            // trace backward
            let mut x = i as f32 - dt0 * velx.get(i, j);
            let mut y = j as f32 - dt0 * vely.get(i, j);

            // clamp to sampling domain
            x = x.clamp(0.5, (n - 1) as f32 + 0.5);
            y = y.clamp(0.5, (n - 1) as f32 + 0.5);

            let i0 = x.floor() as usize;
            let i1 = (i0 + 1).min(n - 1);
            let j0 = y.floor() as usize;
            let j1 = (j0 + 1).min(n - 1);

            let s1 = x - i0 as f32;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f32;
            let t0 = 1.0 - t1;

            let val = s0 * (t0 * d0.get(i0, j0) + t1 * d0.get(i0, j1))
                + s1 * (t0 * d0.get(i1, j0) + t1 * d0.get(i1, j1));
            d.set(i, j, val);
        }
    }

    set_bound(b, d);
}

/// Enforce incompressibility: compute div, solve for pressure p, subtract grad(p)
fn project(velx: &mut Grid, vely: &mut Grid, p: &mut Grid, div: &mut Grid, iter: usize) {
    let n = velx.n;
    let n_recip = 1.0 / n as f32;

    for j in 1..n - 1 {
        for i in 1..n - 1 {
            let divv = -0.5
                * (velx.get(i + 1, j) - velx.get(i - 1, j)
                    + vely.get(i, j + 1) - vely.get(i, j - 1))
                * n_recip;
            div.set(i, j, divv);
            p.set(i, j, 0.0);
        }
    }

    set_bound(BoundType::Scalar, div);
    set_bound(BoundType::Scalar, p);

    // Solve ∇² p = div; here a = 1, c = 4 for 2D Laplacian
    lin_solve(BoundType::Scalar, p, div, 1.0, 4.0, iter);

    for j in 1..n - 1 {
        for i in 1..n - 1 {
            let gx = 0.5 * (p.get(i + 1, j) - p.get(i - 1, j)) * n as f32;
            let gy = 0.5 * (p.get(i, j + 1) - p.get(i, j - 1)) * n as f32;
            velx.set(i, j, velx.get(i, j) - gx);
            vely.set(i, j, vely.get(i, j) - gy);
        }
    }

    set_bound(BoundType::Vx, velx);
    set_bound(BoundType::Vy, vely);
}
