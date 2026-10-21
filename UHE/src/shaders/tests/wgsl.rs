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
    let spirv = wgsl::to_spirv(TRIANGLE, "triangle").expect("triangle should compile");

    assert!(spirv.len() > 20, "{} bytes is not a module", spirv.len());
    assert_eq!(spirv.len() % 4, 0, "SPIR-V is a stream of 32-bit words");

    let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
    assert_eq!(magic, 0x0723_0203, "not a SPIR-V module");
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
    let error = wgsl::to_spirv("fn vs_main() -> { }", "broken").expect_err("this is not a shader");

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

#[test]
fn the_same_source_compiles_to_the_same_bytes() {
    let once = wgsl::to_spirv(TRIANGLE, "triangle").unwrap();
    let twice = wgsl::to_spirv(TRIANGLE, "triangle").unwrap();

    assert_eq!(once, twice);
}

/// The module keeps the names the source declared.
///
/// Slang emits `main` from its own compile of the same source; neither the
/// backends nor `forge` rename one to look like the other. A backend reads
/// `OpEntryPoint` and names the stage with what it finds.
#[test]
fn the_emitted_module_keeps_the_source_names() {
    let spirv = wgsl::to_spirv(TRIANGLE, "triangle").unwrap();

    assert_eq!(
        entry_point_names(&spirv),
        vec!["vs_main".to_owned(), "fs_main".to_owned()]
    );
}

/// Every `OpEntryPoint`'s name, read out of the word stream.
///
/// The instruction is `OpEntryPoint <execution model> <function id> <name>`:
/// the name is the third word onwards, NUL-terminated and padded to a word.
fn entry_point_names(spirv: &[u8]) -> Vec<String> {
    const OP_ENTRY_POINT: u32 = 15;

    let words: Vec<u32> = spirv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();

    let mut names = Vec::new();
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
            names.push(String::from_utf8_lossy(&literal[..end]).into_owned());
        }
        index += count;
    }

    names
}
