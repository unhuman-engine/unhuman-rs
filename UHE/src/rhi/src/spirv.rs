//! SPIR-V entry points.
//!
//! A backend names a pipeline stage's entry point when it builds the stage, and
//! that name belongs to the module: Slang emits `main`, naga emits the source's
//! name, and neither compiler's choice is a convention the other has to imitate.

use crate::desc::ShaderStage;

const HEADER_WORDS: usize = 5;
const MAGIC: u32 = 0x0723_0203;
const OP_ENTRY_POINT: u16 = 15;

/// One `OpEntryPoint`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EntryPoint {
    pub stage: ShaderStage,
    pub name: String,
}

/// Every entry point in a module, in declaration order.
///
/// `None` when `words` is not a SPIR-V module. A module with no entry point
/// yields an empty list; the caller decides whether that is an error.
pub fn entry_points(words: &[u32]) -> Option<Vec<EntryPoint>> {
    if words.len() < HEADER_WORDS || words[0] != MAGIC {
        return None;
    }

    let mut entries = Vec::new();
    let mut index = HEADER_WORDS;
    while index < words.len() {
        let header = words[index];
        let word_count = (header >> 16) as usize;
        let opcode = (header & 0xffff) as u16;
        if word_count == 0 || index + word_count > words.len() {
            return None;
        }

        if opcode == OP_ENTRY_POINT {
            // OpEntryPoint <execution model> <function id> <name> [<interface>...]
            if word_count < 4 {
                return None;
            }
            if let Some(stage) = execution_model(words[index + 1]) {
                entries.push(EntryPoint {
                    stage,
                    name: literal(&words[index + 3..index + word_count]),
                });
            }
        }

        index += word_count;
    }

    Some(entries)
}

fn execution_model(value: u32) -> Option<ShaderStage> {
    Some(match value {
        0 => ShaderStage::Vertex,
        1 => ShaderStage::TessellationControl,
        2 => ShaderStage::TessellationEvaluation,
        3 => ShaderStage::Geometry,
        4 => ShaderStage::Fragment,
        5 => ShaderStage::Compute,
        // Kernel, and any model added after this parser was written: no RHI
        // pipeline stage names them.
        _ => return None,
    })
}

/// A NUL-terminated literal padded to a word boundary.
fn literal(words: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instruction(word_count: u32, opcode: u32) -> u32 {
        (word_count << 16) | opcode
    }

    fn entry_point(model: u32, name: &str) -> Vec<u32> {
        let mut bytes = name.as_bytes().to_vec();
        bytes.push(0);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        let name_words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        let mut words = vec![
            instruction(3 + name_words.len() as u32, u32::from(OP_ENTRY_POINT)),
            model,
            1,
        ];
        words.extend(name_words);
        words
    }

    fn module(body: Vec<u32>) -> Vec<u32> {
        let mut words = vec![MAGIC, 0x0001_0000, 0, 100, 0];
        words.extend(body);
        words
    }

    #[test]
    fn reads_each_stage_from_a_multi_stage_module() {
        let mut body = entry_point(0, "vs_main");
        body.extend(entry_point(4, "fs_main"));

        assert_eq!(
            entry_points(&module(body)),
            Some(vec![
                EntryPoint {
                    stage: ShaderStage::Vertex,
                    name: "vs_main".to_owned(),
                },
                EntryPoint {
                    stage: ShaderStage::Fragment,
                    name: "fs_main".to_owned(),
                },
            ])
        );
    }

    #[test]
    fn keeps_the_modules_own_name() {
        let entries = entry_points(&module(entry_point(5, "main"))).expect("a module");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].stage, ShaderStage::Compute);
        assert_eq!(entries[0].name, "main");
    }

    #[test]
    fn non_spirv_is_none() {
        assert_eq!(entry_points(&[0; 8]), None);
        assert_eq!(entry_points(&module(Vec::new())).map(|e| e.len()), Some(0));
    }
}
