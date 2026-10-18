//! WGSL to SPIR-V, end to end, with no device.

use forge::wgsl;
use forge::Stage;

const TRIANGLE: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.0,  0.5),
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    out.color = vec3<f32>(1.0, 0.0, 0.0);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

/// A module is bytes, and the magic number is the first word: a caller that
/// gets an empty or text buffer would fail at pipeline creation with nothing
/// pointing back here.
#[test]
fn a_shader_compiles_to_a_spirv_module() {
    let spirv = wgsl::to_spirv_entry(TRIANGLE, "triangle", "vs_main", Stage::Vertex)
        .expect("vertex shader should compile");

    assert!(spirv.len() > 20, "{} bytes is not a module", spirv.len());
    assert_eq!(spirv.len() % 4, 0, "SPIR-V is a stream of 32-bit words");

    let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
    assert_eq!(magic, 0x0723_0203, "not a SPIR-V module");
}

#[test]
fn each_stage_compiles_separately() {
    let vertex = wgsl::to_spirv_entry(TRIANGLE, "triangle", "vs_main", Stage::Vertex).unwrap();
    let fragment = wgsl::to_spirv_entry(TRIANGLE, "triangle", "fs_main", Stage::Fragment).unwrap();

    assert_ne!(vertex, fragment);
}

#[test]
fn the_entry_points_are_named_not_guessed() {
    let entries = wgsl::entry_points(TRIANGLE, "triangle").unwrap();

    assert_eq!(
        entries,
        vec![
            ("vs_main".to_owned(), Stage::Vertex),
            ("fs_main".to_owned(), Stage::Fragment),
        ]
    );
}

/// A syntax error has to be reported with the label of the shader, because the
/// caller's next step is to open that file.
#[test]
fn a_broken_shader_names_itself() {
    let error = wgsl::to_spirv_entry("fn vs_main() -> { }", "broken", "vs_main", Stage::Vertex)
        .expect_err("this is not a shader");

    let message = error.to_string();
    assert!(message.starts_with("broken"), "{message}");
}

/// Type errors are what the validator is for: the parser is happy here.
#[test]
fn validation_catches_what_the_parser_accepts() {
    let bad = r#"
@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, "not a float");
}
"#;

    assert!(wgsl::check(bad, "bad-types").is_err());
}

/// SPIR-V has three stages. Asking for a mesh shader is refused by name rather
/// than compiled into something the driver will reject later.
#[test]
fn a_stage_without_a_spirv_equivalent_is_refused() {
    let error = wgsl::to_spirv_entry(TRIANGLE, "triangle", "vs_main", Stage::Mesh)
        .expect_err("no SPIR-V mesh stage");

    assert!(error.to_string().contains("triangle"), "{error}");
}

#[test]
fn the_same_source_compiles_to_the_same_bytes() {
    let once = wgsl::to_spirv_entry(TRIANGLE, "triangle", "vs_main", Stage::Vertex).unwrap();
    let twice = wgsl::to_spirv_entry(TRIANGLE, "triangle", "vs_main", Stage::Vertex).unwrap();

    assert_eq!(once, twice);
}

/// The module's entry point is named `main`, whatever the source called it.
///
/// The backends name the entry point when they build a pipeline stage, and a
/// module that kept the source name (`vs_main`) creates no pipeline on Vulkan:
/// the driver reports it as a shader error, which points at the wrong file.
#[test]
fn the_emitted_module_names_its_entry_point_main() {
    let vertex = wgsl::to_spirv_entry(TRIANGLE, "triangle", "vs_main", Stage::Vertex).unwrap();
    let fragment = wgsl::to_spirv_entry(TRIANGLE, "triangle", "fs_main", Stage::Fragment).unwrap();

    assert_eq!(entry_point_name(&vertex), forge::ENTRY_POINT);
    assert_eq!(entry_point_name(&fragment), forge::ENTRY_POINT);
}

/// The first `OpEntryPoint`'s name, read out of the word stream.
///
/// The instruction is `OpEntryPoint <execution model> <function id> <name>`:
/// the name is the third word onwards, NUL-terminated and padded to a word.
fn entry_point_name(spirv: &[u8]) -> String {
    const OP_ENTRY_POINT: u32 = 15;

    let words: Vec<u32> = spirv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();

    let mut index = 5;
    while index < words.len() {
        let word = words[index];
        let (count, opcode) = ((word >> 16) as usize, word & 0xffff);
        if opcode == OP_ENTRY_POINT {
            let literal: Vec<u8> = words[index + 3..index + count]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            let end = literal
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(literal.len());
            return String::from_utf8_lossy(&literal[..end]).into_owned();
        }
        index += count;
    }

    panic!("the module has no OpEntryPoint");
}
