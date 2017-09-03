#[macro_use]
extern crate gfx;
extern crate gfx_window_glutin;
extern crate glutin;
extern crate image;
extern crate byteorder;

use gfx::Device;
use gfx::traits::Factory;
use gfx::traits::FactoryExt;
use glutin::GlContext;

pub type ColorFormat = gfx::format::Rgba8;
pub type DepthFormat = gfx::format::DepthStencil;

use std::error::Error;

gfx_defines! {
    vertex Vertex {
        pos: [f32; 2] = "a_Pos",
        tex_pos: [f32; 2] = "a_TexPos",
    }

    constant Transform {
        transform: [[f32; 4]; 4] = "u_Transform",
    }

    constant Variants {
        variant:        [i32; 4] = "u_variant",
        block_size:     [f32; 2] = "u_BlockSize",
        tex_block_size: [f32; 2] = "u_TexBlockSize",
        discard:         u32     = "u_discard",
    }

    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        transform: gfx::ConstantBuffer<Transform> = "Transform",
        variants: gfx::ConstantBuffer<Variants> = "VariantData",
        tiled_tex: gfx::TextureSampler<[f32; 4]> = "t_Texture",
        tile_map: gfx::TextureSampler<[f32; 4]> = "t_Tiles",
        out: gfx::BlendTarget<ColorFormat> = ("Target0", gfx::state::MASK_ALL, gfx::preset::blend::ALPHA),
    }
}

const VERTEX_SHADER: &[u8] = b"
#version 330 core

in vec2 a_Pos;
in vec2 a_TexPos;

uniform Transform {
    mat4 u_Transform;
};

out vec2 v_TexPos;

void main() {
    v_TexPos = a_TexPos;
    gl_Position = u_Transform * vec4(a_Pos, 0.0, 1.0);
}
";

const PIXEL_SHADER: &[u8] = b"
#version 330 core

in vec2 v_TexPos;

uniform VariantData {
    ivec4 u_variant;
    vec2  u_BlockSize;
    vec2  u_TexBlockSize;
    uint  u_discard;
};

uniform sampler2D t_Texture;
uniform sampler1D t_Tiles;
uniform sampler1D t_Alpha;

out vec4 Target0;

vec2 find_block(int start, int count, ivec2 BlockIndex) {
    // simple bisection sort
    int end = start + count - 1;
    for (int i = 0; i < 16; i++) {
        if (start > end) {
            break;
        }
        int middle = (start + end) / 2;

        vec4 sample = texelFetch(t_Tiles, middle, 0);
        ivec2 sampleIndex = ivec2(int(round(sample.x)), int(round(sample.y)));

        if (sampleIndex.y > BlockIndex.y || (sampleIndex.y == BlockIndex.y && sampleIndex.x > BlockIndex.x)) {
            end = middle - 1;
        } else if (sampleIndex.y < BlockIndex.y || (sampleIndex.y == BlockIndex.y && sampleIndex.x < BlockIndex.x))  {
            start = middle + 1;
        } else {
            return sample.zw;
        }
    }
    return vec2(-1.0);

}

void main() {
    // get the block index
    vec2 BlockIndexF = vec2(0.0);
    vec2 BlockPos = modf(v_TexPos / u_BlockSize, BlockIndexF);

    // deal with negative numbers.
    if (sign(BlockPos.x) < 0) {
        BlockIndexF.x -= 1;
        BlockPos.x = 1 + BlockPos.x;
    }
    if (sign(BlockPos.y) < 0) {
        BlockIndexF.y -= 1;
        BlockPos.y = 1 + BlockPos.y;
    }

    // convert the index to a set of ints for the lookup
    ivec2 BlockIndex = ivec2(int(round(BlockIndexF.x)), int(round(BlockIndexF.y)));

    vec2 BlockTexPos = find_block(u_variant.x, u_variant.y, BlockIndex);

    // if we didn't find something, discard this fragment.
    if (BlockTexPos.x < 0.0) {
        if (u_discard != 0u) {
            discard;
        }
        Target0 = vec4(BlockPos.x, 0.0, BlockPos.y, 1.0);
        return;
    }

    vec2 texPos = BlockTexPos + BlockPos * u_TexBlockSize;
    vec4 color = texture(t_Texture, texPos);

    Target0 = pow(color, vec4(1.4));
}

";

// 3D identity matrix
static TRANSFORM: [[f32; 4]; 4] = 
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]];

    // [[0.7, 0.2, 0.0, 0.2],
    //  [-0.2, 0.7, 0.0, -0.5],
    //  [0.0, 0.0, 1.0, 0.0],
    //  [0.0, 0.0, 0.0, 1.0]];

const CLEAR_COLOR: [f32; 4] = [0.5, 0.5, 0.5, 1.0];

// boiler plate error hanlding
pub fn main() {
    if let Err(e) = console_main() {
        println!("{}", e);
    }
}

// handle args
fn console_main() -> Result<(), Box<Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("Expected two argument: texture packing".into());
    }

    let tex_file     = &args[1];
    let tilemap_file = &args[2];

    render(tilemap_file, tex_file)
}

pub fn render(tilemap_file: &str, tex_file: &str) -> Result<(), Box<Error>> {
    // load data
    let sprites = SpriteVariants::new(
        tilemap_file,
        tex_file
    )?;
    let (width, height) = sprites.size();

    // window creation functions.
    let mut events_loop = glutin::EventsLoop::new();
    let builder = glutin::WindowBuilder::new()
        .with_title("Sprite rendering example")
        .with_dimensions(width as u32, height as u32);
    let context = glutin::ContextBuilder::new()
        .with_gl_robustness(glutin::Robustness::TryRobustLoseContextOnReset)
        .with_srgb(true)
        .with_multisampling(4)
        .with_vsync(true);
    let (window, mut device, mut factory, main_color, main_depth) =
        gfx_window_glutin::init::<ColorFormat, DepthFormat>(builder, context, &events_loop);

    // pipeline
    let pso = factory.create_pipeline_simple(VERTEX_SHADER, PIXEL_SHADER, pipe::new())?;

    // create a quad
    let quad = [
        Vertex { pos: [ -1.0, -1.0], tex_pos: [         0.0, height as f32] },
        Vertex { pos: [  1.0, -1.0], tex_pos: [width as f32, height as f32] },
        Vertex { pos: [ -1.0,  1.0], tex_pos: [         0.0,           0.0] },
        Vertex { pos: [  1.0,  1.0], tex_pos: [width as f32,           0.0] },
        Vertex { pos: [ -1.0,  1.0], tex_pos: [         0.0,           0.0] },
        Vertex { pos: [  1.0, -1.0], tex_pos: [width as f32, height as f32] }
    ];

    // tiled texture
    let img = sprites.texture();
    let (width, height) = img.dimensions();
    let kind = gfx::texture::Kind::D2(width as u16, height as u16, gfx::texture::AaMode::Single);
    let (_, tiled_tex) = factory.create_texture_immutable_u8::<gfx::format::Rgba8>(kind, &[&img])?;

    // data
    let mut tile_variant = 0;
    let mut discard = true;

    let tile_data = sprites.get_tile_data(tile_variant)?;
    // necessary evil
    let temp_tile_data = unsafe {
        std::slice::from_raw_parts(tile_data.as_ptr() as *const [u32; 4], tile_data.len())
    };
    let kind = gfx::texture::Kind::D1(tile_data.len() as u16);
    let (_, tile_map) = factory.create_texture_immutable::<gfx::format::Rgba32F>(kind, &[temp_tile_data])?;


    // pipeline data
    let (vertex_buffer, slice) = factory.create_vertex_buffer_with_slice(&quad, ());
    let transform_buffer = factory.create_constant_buffer(1);
    let variant_buffer = factory.create_constant_buffer(1);
    let mut data = pipe::Data {
        vbuf:       vertex_buffer,
        transform:  transform_buffer,
        variants:   variant_buffer,
        tiled_tex:  (tiled_tex,  factory.create_sampler_linear()),
        tile_map:   (tile_map,   factory.create_sampler_linear()),
        out:        main_color.clone()
    };

    // move data to GPU
    let mut encoder: gfx::Encoder<_, _> = factory.create_command_buffer().into();
    encoder.update_constant_buffer(&data.transform, &Transform {
        transform: TRANSFORM
    });
    encoder.update_constant_buffer(&data.variants, &Variants {
        variant: [0, tile_data.len() as i32, 0, 0],
        block_size: sprites.block_size(),
        tex_block_size: sprites.texture_block_size(),
        discard: discard as u32
    });
    encoder.flush(&mut device);

    // render loop
    let mut running = true;
    while running {
        // render something
        encoder.clear(&main_color, CLEAR_COLOR);
        encoder.clear_depth(&main_depth, 1.0);
        encoder.draw(&slice, &pso, &data);
        encoder.flush(&mut device);
        window.swap_buffers()?;
        device.cleanup();

        let old_tile_variant = tile_variant;
        let old_discard = discard;

        // handle events
        events_loop.poll_events(|event| {
            if let glutin::Event::WindowEvent {event, ..} = event {
                match event {
                    glutin::WindowEvent::Closed => running = false,
                    glutin::WindowEvent::Resized(w, h) => {
                        window.resize(w, h)
                    },
                    glutin::WindowEvent::KeyboardInput {
                        input: glutin::KeyboardInput {virtual_keycode: Some(key), state: glutin::ElementState::Pressed, ..}, ..
                    } => match key {
                        glutin::VirtualKeyCode::Escape => running = false,
                        glutin::VirtualKeyCode::Right => {
                            if tile_variant + 1 < sprites.variants() {
                                tile_variant += 1;
                            }
                        },
                        glutin::VirtualKeyCode::Left => {
                            if tile_variant != 0 {
                                tile_variant -= 1;
                            }
                        },
                        glutin::VirtualKeyCode::Down
                        | glutin::VirtualKeyCode::Up => {
                            discard = !discard;
                        },
                        v => println!("{:?}", v)
                    },
                    _ => {}
                }
            }
        });

        if old_tile_variant != tile_variant || old_discard != discard {
            let tile_data = sprites.get_tile_data(tile_variant)?;
            // necessary evil
            let temp_tile_data = unsafe {
                std::slice::from_raw_parts(tile_data.as_ptr() as *const [u32; 4], tile_data.len())
            };
            let kind = gfx::texture::Kind::D1(tile_data.len() as u16);
            let (_, tile_map) = factory.create_texture_immutable::<gfx::format::Rgba32F>(kind, &[temp_tile_data])?;

            data.tile_map.0 = tile_map;

            encoder.update_constant_buffer(&data.variants, &Variants {
                variant: [0, tile_data.len() as i32, 0, 0],
                block_size: sprites.block_size(),
                tex_block_size: sprites.texture_block_size(),
                discard: discard as u32
            });
        }

    }

    Ok(())
}


use std::collections::HashMap;
use std::fs::File;
use byteorder::{ReadBytesExt, LittleEndian};

#[derive(Debug, Clone, Copy)]
enum Variant {
    Base(u8),
    Single(u8),
    Double(u8, u8)
}

#[derive(Debug, Clone)]
struct SpriteVariants {
    pub variants:  Vec<(Variant, usize, usize)>,
    pub blocks:    Vec<[f32; 4]>,
    pub block_map: HashMap<u8, (usize, usize)>,
    pub texture:   image::RgbaImage,
    pub size:      (usize, usize)
}

impl SpriteVariants{
    fn new(scramble_path: &str, image_path: &str) -> Result<Self, Box<Error>> {
        let mut scramble_file = File::open(scramble_path)?;

        let variant_count = scramble_file.read_u32::<LittleEndian>()?;
        let block_count   = scramble_file.read_u32::<LittleEndian>()?;

        let mut variants = Vec::new();
        for _ in 0 .. variant_count {
            let parent1 = scramble_file.read_u8()?;
            let parent2 = scramble_file.read_u8()?;
            let parent3 = scramble_file.read_u8()?;
            assert!(parent3 == 0x00);
            let instruction = scramble_file.read_u8()?;

            let block_offset = scramble_file.read_u32::<LittleEndian>()? as usize;
            let block_count  = scramble_file.read_u32::<LittleEndian>()? as usize;

            variants.push((
                match instruction {
                    0x00 => Variant::Base(parent1),
                    0x20 => Variant::Single(parent1),
                    0x40 => Variant::Double(parent1, parent2),
                    _ => return Err("Unknown format".into())
                },
                block_offset,
                block_offset + block_count
            ));
        }
        
        let mut blocks = Vec::new();
        for _ in 0 .. block_count {
            let mut temp = [0.0f32; 4];
            temp[0] = scramble_file.read_f32::<LittleEndian>()?;
            temp[1] = scramble_file.read_f32::<LittleEndian>()?;
            temp[2] = scramble_file.read_f32::<LittleEndian>()?;
            temp[3] = scramble_file.read_f32::<LittleEndian>()?;
            blocks.push(temp);
        }

        let mut block_map = HashMap::new();
        for &(variant, start, end) in variants.iter() {
            match variant {
                Variant::Base(id) => {
                    block_map.insert(0,  (start, end));
                    block_map.insert(id, (start, end));
                },
                Variant::Single(id) => {
                    block_map.insert(id, (start, end));
                },
                Variant::Double(_, _) => ()
            }
        }

        // comparison func that just says things are equal when NaNs are present
        use std::cmp::Ordering;
        let float_cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(Ordering::Equal);

        // change the coordinates into our slightly less silly coord system
        let min_x = blocks.iter().map(|arr| arr[0]).min_by(&float_cmp).unwrap_or(0.0);
        let max_x = blocks.iter().map(|arr| arr[0]).max_by(&float_cmp).unwrap_or(0.0) + 30.0;

        let min_y = blocks.iter().map(|arr| arr[1]).min_by(&float_cmp).unwrap_or(0.0);
        let max_y = blocks.iter().map(|arr| arr[1]).max_by(&float_cmp).unwrap_or(0.0) + 30.0;

        for block in blocks.iter_mut() {
            block[0] = ((block[0] - min_x) / 30.0).round();
            block[1] = ((block[1] - min_y) / 30.0).round();
        }

        let size = ((max_x - min_x).round() as usize, (max_y - min_y).round() as usize);

        Ok(SpriteVariants {
            variants: variants,
            blocks: blocks,
            block_map: block_map,
            texture: image::open(image_path)?.to_rgba(),
            size: size
        })
    }

    fn size(&self) -> (usize, usize) {
        self.size
    }

    fn block_size(&self) -> [f32; 2] {
        [30.0, 30.0]
    }

    fn texture_block_size(&self) -> [f32; 2] {
        let (width, height) = self.texture.dimensions();
        [30.0 / width as f32, 30.0 / height as f32]
    }

    fn texture(&self) -> &image::RgbaImage {
        &self.texture
    }

    fn variants(&self) -> usize {
        self.variants.len()
    }

    fn get_tile_data(&self, variant: usize) -> Result<Vec<[f32; 4]>, Box<Error>> {
        let &(variant, start, end) = self.variants.get(variant).ok_or("Unknown variant")?;
        let blocks = match variant {
            Variant::Base(_) => {
                self.blocks[start .. end].to_owned()
            },
            Variant::Single(_) => {
                let (start_z, end_z) = self.block_map[&0];
                SortedMerge::new(self.blocks[start_z .. end_z].iter().cloned(), self.blocks[start .. end].iter().cloned()).collect()
            },
            Variant::Double(ref id1, ref id2) => {
                let (start_z, end_z) = self.block_map[&0];
                let &(start1, end1) = self.block_map.get(id1).unwrap_or(&(0, 0));
                let &(start2, end2) = self.block_map.get(id2).unwrap_or(&(0, 0));
                let temp = SortedMerge::new(self.blocks[start_z .. end_z].iter().cloned(), self.blocks[start1 .. end1].iter().cloned());
                let temp = SortedMerge::new(temp, self.blocks[start2 .. end2].iter().cloned());
                SortedMerge::new(temp, self.blocks[start .. end].iter().cloned()).collect()
            }
        };

        Ok(blocks)
    }
}

use std::iter::Peekable;

struct SortedMerge<A, B>
    where A: Iterator<Item=[f32; 4]>, B: Iterator<Item=[f32; 4]> {
    left: Peekable<A>,
    right: Peekable<B>
}

impl<A, B> SortedMerge<A, B>
    where A: Iterator<Item=[f32; 4]>, B: Iterator<Item=[f32; 4]> {

    fn new<C, D>(left: C, right: D) -> SortedMerge<A, B>
        where C: IntoIterator<IntoIter=A, Item=[f32; 4]>, D: IntoIterator<IntoIter=B, Item=[f32; 4]> {

        SortedMerge {
            left: left.into_iter().peekable(),
            right: right.into_iter().peekable()
        }
    }

}

impl<A, B> Iterator for SortedMerge<A, B>
    where A: Iterator<Item=[f32; 4]>, B: Iterator<Item=[f32; 4]> {

    type Item = [f32; 4];

    fn next(&mut self) -> Option<Self::Item> {
        match (self.left.peek().cloned(), self.right.peek().cloned()) {
            (None,    None   ) => None,
            (None,    Some(_)) => self.right.next(),
            (Some(_), None   ) => self.left.next(),
            (Some(l), Some(r)) => {
                if l[1] > r[1] || (l[1] == r[1] && l[0] > r[0]) {
                    self.right.next()
                } else if l[1] < r[1] || (l[1] == r[1] && l[0] < r[0]) {
                    self.left.next()
                } else {
                    self.left.next();
                    self.right.next()
                }
            }
        }
    }
}
