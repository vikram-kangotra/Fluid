// src/main.rs
mod grid;
mod fluid;

use crate::fluid::Fluid;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::PixelFormatEnum;

const SIZE: usize = 128;
const SCALE: usize = 8;
const ITER: usize = 16;

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline(always)]
fn lerp_color(c1: (u8, u8, u8), c2: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    (
        lerp(c1.0 as f32, c2.0 as f32, t) as u8,
        lerp(c1.1 as f32, c2.1 as f32, t) as u8,
        lerp(c1.2 as f32, c2.2 as f32, t) as u8,
    )
}

fn color_from_density_speed(density: f32, vx: f32, vy: f32) -> (u8, u8, u8) {
    // density [0..inf] => brightness
    let brightness = (density * 255.0).clamp(0.0, 255.0);
    let density_weight = (brightness / 255.0).powf(0.8);

    let speed = vx.hypot(vy);
    let speed_norm = (speed * 50.0).clamp(0.0, 255.0) / 255.0;
    let speed_weight = speed_norm.powf(0.6);

    let base_r = 200.0 * density_weight;
    let base_g = 200.0 * density_weight;
    let base_b = 220.0 * density_weight;

    let tint_r = 255.0 * speed_weight;
    let tint_g = 120.0 * speed_weight;
    let tint_b = 40.0 * speed_weight;

    let r = (base_r + tint_r).clamp(0.0, 255.0) as u8;
    let g = (base_g + tint_g).clamp(0.0, 255.0) as u8;
    let b = (base_b + tint_b).clamp(0.0, 255.0) as u8;

    (r, g, b)
}

fn main() -> Result<(), String> {
    // SDL2 init
    let sdl_context = sdl2::init()?;
    let video = sdl_context.video()?;
    let window = video
        .window(
            "Stable Fluids — refactor",
            (SIZE * SCALE) as u32,
            (SIZE * SCALE) as u32,
        )
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().present_vsync().build().map_err(|e| e.to_string())?;
    let tc = canvas.texture_creator();
    let mut texture = tc
        .create_texture_streaming(PixelFormatEnum::RGB24, (SIZE * SCALE) as u32, (SIZE * SCALE) as u32)
        .map_err(|e| e.to_string())?;

    let mut event_pump = sdl_context.event_pump()?;

    // Simulation params and stability heuristics
    let max_speed = 1.0_f32;
    let dx = 1.0 / SIZE as f32;
    let diffusion = 0.0005_f32;
    let viscosity = 0.0001_f32;

    let advec_limit = dx / max_speed;
    let diff_limit = dx * dx / (4.0 * diffusion);
    let visc_limit = dx * dx / (4.0 * viscosity);
    let dt = 0.5 * advec_limit.min(diff_limit).min(visc_limit);

    let mut cube = Fluid::new(SIZE, diffusion, viscosity, dt);

    let mut running = true;
    let mut mouse_down = false;
    let mut last_mouse = (0usize, 0usize);

    // Preallocated buffers
    let mut pixels = vec![0u8; SIZE * SCALE * SIZE * SCALE * 3];
    let mut color_grid = vec![(0u8, 0u8, 0u8); SIZE * SIZE];

    // Main loop
    while running {

        // input
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => running = false,

                Event::MouseButtonDown {
                    mouse_btn: MouseButton::Left,
                    x,
                    y,
                    ..
                } => {
                    mouse_down = true;
                    last_mouse = (x as usize, y as usize);
                }

                Event::MouseButtonUp {
                    mouse_btn: MouseButton::Left,
                    ..
                } => mouse_down = false,

                Event::MouseMotion { x, y, .. } => {
                    if mouse_down {
                        let mx = (x as usize / SCALE).clamp(0, SIZE - 1);
                        let my = (y as usize / SCALE).clamp(0, SIZE - 1);

                        // you used (y,x) earlier; here we keep (x,y) => ensure consistency when reading grid
                        cube.add_density(mx, my, 50.0 * dt / 0.001);

                        let amtx = (x as i32 - last_mouse.0 as i32) as f32;
                        let amty = (y as i32 - last_mouse.1 as i32) as f32;
                        // Add velocity with some scaling
                        cube.add_velocity(mx, my, amtx * 10.0, amty * 10.0);
                    }
                    last_mouse = (x as usize, y as usize);
                }
                _ => {}
            }
        }

        // physics
        cube.step(ITER);

        // compute per-cell color (density + velocity tint)
        for y in 0..SIZE {
            for x in 0..SIZE {
                let d = cube.density.get(x, y);
                let vx = cube.vx.get(x, y);
                let vy = cube.vy.get(x, y);
                color_grid[y * SIZE + x] = color_from_density_speed(d, vx, vy);
            }
        }

        // upscale with bilinear interpolation from color_grid -> pixels
        let row_stride = SIZE * SCALE * 3;
        for y in 0..(SIZE * SCALE) {
            let gy = y as f32 / SCALE as f32;
            let y0 = gy.floor() as usize;
            let y1 = (y0 + 1).min(SIZE - 1);
            let ty = gy - y0 as f32;

            for x in 0..(SIZE * SCALE) {
                let gx = x as f32 / SCALE as f32;
                let x0 = gx.floor() as usize;
                let x1 = (x0 + 1).min(SIZE - 1);
                let tx = gx - x0 as f32;

                let c00 = color_grid[y0 * SIZE + x0];
                let c10 = color_grid[y0 * SIZE + x1];
                let c01 = color_grid[y1 * SIZE + x0];
                let c11 = color_grid[y1 * SIZE + x1];

                let cx0 = lerp_color(c00, c10, tx);
                let cx1 = lerp_color(c01, c11, tx);
                let (r, g, b) = lerp_color(cx0, cx1, ty);

                let idx = y * row_stride + x * 3;
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
            }
        }

        texture.update(None, &pixels, row_stride).unwrap();
        canvas.clear();
        canvas.copy(&texture, None, None)?;
        canvas.present();
    }

    Ok(())
}
