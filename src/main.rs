mod fluid_cube;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::PixelFormatEnum;
use crate::fluid_cube::FluidCube;

const SIZE: usize = 128;
const SCALE: usize = 8; // render scaling factor
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

fn main() -> Result<(), String> {
    // --- SDL setup ---
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window("SDL2 Smoke Simulation (Optimized + Smooth)", (SIZE * SCALE) as u32, (SIZE * SCALE) as u32)
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().present_vsync().build().map_err(|e| e.to_string())?;
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::RGB24, (SIZE * SCALE) as u32, (SIZE * SCALE) as u32)
        .map_err(|e| e.to_string())?;

    let mut event_pump = sdl_context.event_pump()?;


    let max_speed = 1.0; // estimated max velocity magnitude (tuned per scene)
    let dx = 1.0 / SIZE as f32;
    let diffusion = 0.0005;
    let viscosity = 0.0001;

    // stability limits
    let advec_limit = dx / max_speed;
    let diff_limit = dx * dx / (4.0 * diffusion);
    let visc_limit = dx * dx / (4.0 * viscosity);

    // choose safe timestep
    let dt = 0.5 * advec_limit.min(diff_limit).min(visc_limit);

    println!("dt={:.6}, advec_limit={:.6}, diff_limit={:.6}, visc_limit={:.6}", dt, advec_limit, diff_limit, visc_limit);

    let mut cube = FluidCube::new(SIZE, diffusion, viscosity, dt);


    let mut running = true;
    let mut mouse_down = false;
    let mut last_mouse_pos = (0usize, 0usize);

    // Preallocate buffers
    let mut pixels = vec![0u8; SIZE * SCALE * SIZE * SCALE * 3];
    let mut color_grid = vec![(0u8, 0u8, 0u8); SIZE * SIZE];

    while running {
        // --- Input ---
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => running = false,
                Event::MouseButtonDown { mouse_btn: MouseButton::Left, .. } => mouse_down = true,
                Event::MouseButtonUp { mouse_btn: MouseButton::Left, .. } => mouse_down = false,
                Event::MouseMotion { x, y, .. } => {
                    if mouse_down {
                        let mx = (x as usize / SCALE).clamp(0, SIZE - 1);
                        let my = (y as usize / SCALE).clamp(0, SIZE - 1);
                        cube.add_density(my, mx, 50.0 * dt / 0.001);
                        let amtx = (x as i32 - last_mouse_pos.0 as i32) as f32;
                        let amty = (y as i32 - last_mouse_pos.1 as i32) as f32;
                        cube.add_velocity(my, mx, amty * 100.0, amtx * 100.0);
                    }
                    last_mouse_pos = (x as usize, y as usize);
                }
                _ => {}
            }
        }

        // --- Physics step ---
        cube.step(ITER);

        // --- Precompute color grid (per simulation cell) ---


        for y in 0..SIZE {
            for x in 0..SIZE {
                let density = cube.density[y][x];
                let speed = cube.vx[y][x].hypot(cube.vy[y][x]);

                // Base brightness from density
                let brightness = (density * 255.0).clamp(0.0, 255.0);
                let density_weight = (brightness / 255.0).powf(0.8);

                // Velocity-based tint (more orange than blue)
                let speed_norm = (speed * 50.0).clamp(0.0, 255.0) / 255.0;
                let speed_weight = speed_norm.powf(0.6);

                // Base gray smoke
                let base_r = 200.0 * density_weight;
                let base_g = 200.0 * density_weight;
                let base_b = 220.0 * density_weight;

                // Orange-blue tint (reduce blue to avoid purple)
                let tint_r = 255.0 * speed_weight;    // warm highlight
                let tint_g = 120.0 * speed_weight;    // subtle green for orange
                let tint_b = 40.0 * speed_weight;     // very little blue

                // Final color
                let r = (base_r + tint_r).clamp(0.0, 255.0) as u8;
                let g = (base_g + tint_g).clamp(0.0, 255.0) as u8;
                let b = (base_b + tint_b).clamp(0.0, 255.0) as u8;

                color_grid[y * SIZE + x] = (r, g, b);
            }
        }


        // --- Bilinear interpolation for smooth upscaling ---
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

        // --- Upload to texture ---
        texture.update(None, &pixels, row_stride).unwrap();

        // --- Render ---
        canvas.clear();
        canvas.copy(&texture, None, None)?;
        canvas.present();
    }

    Ok(())
}
