# Crate map and seams

The port of the C++ tree (`../UNHUMAN`) into this workspace. `docs/architecture/00-spine.md`
is the design; this file is the interface contract every crate is written against, so
crates can be written in parallel without reading each other's code.

**Rules for this port**

1. Idiomatic, safe Rust. `unsafe` is a last resort and needs a written argument. No
   ambient globals: no `static mut`, no `OnceCell` service locator, no `extern` — the
   C++ tree's five failure modes in `00-spine.md` §0 are the whole reason for the port.
2. Every crate inherits `edition` from the workspace (2021, the edition the existing
   crates were written in; the port does not spend its one chance to be right on an
   edition bump) and gets `cargo fmt`-clean code.
3. Tests are written, not run: each crate carries `tests/*.rs` against its public API.
4. **Nothing is verified by compiling.** No agent runs `cargo build`, `cargo check`,
   `cargo test` or `cargo clippy`. Read the dependency's source in
   `~/.cargo/registry/src/*/` before calling into it. Type-level care replaces the
   compiler here, so prefer the boring, spelled-out form over a clever one.
5. No new abstractions "for later". A type exists because a caller names it.

## Layout

```
UHE/
  src/core/          adam      Engine, App, Plugin, FrameCtx, Time, backend traits
  src/engine/        engine    the one crate that sees both backends: the GPU factory
  src/rhi/           chassis   RHI + render graph (existing)
  src/jobs/          grind     job system + task graph (existing)
  src/shaders/       forge     Slang/WGSL in, SPIR-V out (existing + wgsl module)
  src/events/        pings     event bus, key/mouse codes, events          [new]
  src/platform/      eyes      window + input (winit)                      [new]
  src/scene/         meats     ECS, components, hierarchy, serialisation   [new]
  src/assets/        vault     asset registry, VFS, loaders                [new]
  src/renderer/      bifrost   renderer frontend, as a Plugin              [new]
  src/audio/         wire      audio backend, as a Plugin                  [new]
  src/physics/       femur     physics backend, as a Plugin                [new]
  Stormbreaker/vulkan/  stormbreaker   Vulkan backend (existing, untouched)
  Stormbreaker/wgpu/    stormglass     wgpu backend (existing; windowed path added)
  bins/nocap/        nocap     headless binary (gpu: None)
  bins/sandbox/      sandbox   windowed binary
  platform/          (docs only)

UHE-EDITOR/                    the editor, kept beside the engine
  mirror/            mirror    editor model, toolkit-agnostic, as a Plugin [new]
  editor/            editor    GPUI editor binary
UHE-AIMLAB/aimlab/   aimlab    the UHEGAME demo layer, ported as a Plugin
```

Dependency direction: `pings` ← `eyes`/`vault` ← `meats` ← `adam` ← {`bifrost`, `wire`,
`femur`, `mirror`} ← bins. `bifrost` depends on `chassis` and never on a backend crate.
No backend crate depends on any of the above. `engine` is the single exception in the other
direction: it sees both backends, and its job is the GPU factory and the parity test that
proves they behave the same through `Box<dyn Gpu>`.

## Seams

Versions come from the root `Cargo.toml`; use `{ workspace = true }` and nothing else.

### `pings` — events

```rust
pub enum KeyCode { A, B, /* … Z, */ Num0, /* … */ Space, Enter, Escape, Tab, Backspace,
                   Left, Right, Up, Down, LeftShift, RightShift, LeftControl, RightControl,
                   LeftAlt, RightAlt, F1, /* … F12 */ Unknown(u32) }
pub enum MouseButton { Left, Right, Middle, Other(u8) }

pub enum Event {
    WindowResized { width: u32, height: u32 },
    WindowCloseRequested,
    WindowFocused(bool),
    KeyPressed { key: KeyCode, repeat: bool },
    KeyReleased { key: KeyCode },
    TextInput(String),
    MouseButtonPressed(MouseButton),
    MouseButtonReleased(MouseButton),
    MouseMoved { x: f32, y: f32 },
    MouseScrolled { dx: f32, dy: f32 },
    FrameBegin,
    FrameEnd,
}

pub struct EventBus { /* VecDeque<Event> + subscribers */ }
impl EventBus {
    pub fn new() -> Self;
    pub fn push(&mut self, event: Event);
    pub fn drain(&mut self) -> Vec<Event>;      // hands the frame's events over
    pub fn subscribe(&mut self, f: impl FnMut(&Event) + Send + 'static) -> Subscription;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

### `eyes` — window and input

```rust
pub struct WindowDesc { pub title: String, pub width: u32, pub height: u32, pub vsync: bool }
pub struct Input { /* held/pressed/released sets, cursor, deltas, scroll */ }
impl Input {
    pub fn new() -> Self;
    pub fn begin_frame(&mut self);              // pressed/released/deltas are per frame
    pub fn end_frame(&mut self);
    pub fn apply(&mut self, event: &pings::Event);   // bus -> state, one source of truth
    pub fn is_key_held(&self, key: pings::KeyCode) -> bool;
    pub fn is_key_pressed(&self, key: pings::KeyCode) -> bool;
    pub fn is_key_released(&self, key: pings::KeyCode) -> bool;
    pub fn is_mouse_held(&self, button: pings::MouseButton) -> bool;
    pub fn is_mouse_pressed(&self, button: pings::MouseButton) -> bool;
    pub fn cursor(&self) -> (f32, f32);
    pub fn cursor_delta(&self) -> (f32, f32);
    pub fn scroll(&self) -> (f32, f32);
}

/// The window is `winit`'s, shared, and *re-exported*: the wgpu backend is handed
/// `Arc<Window>` and calls `Instance::create_surface(window.clone())` itself, so no
/// backend crate has to name a windowing library (`raw-window-handle` 0.6 comes in
/// with winit and is what wgpu 30 matches).
pub use winit::window::Window;
impl Window {
    pub fn size(&self) -> (u32, u32);
    pub fn scale_factor(&self) -> f32;
    pub fn set_title(&self, title: &str);
    pub fn request_redraw(&self);
    pub fn request_close(&self);
}

/// The event loop owns the process: winit 0.30 runs the loop, so the frame is a
/// callback, not a `while` we control. Headless never constructs a `Window`.
pub fn run_windowed(desc: WindowDesc, frame: impl FnMut(&mut FrameView<'_>) -> bool + 'static)
    -> Result<(), WindowError>;

pub struct FrameView<'a> {
    pub window: &'a std::sync::Arc<Window>,
    pub events: &'a [pings::Event],
    pub elapsed: std::time::Duration,
}
```

`FrameView` carries no `Input`. `adam::Engine::input` is the single instance and
`App::frame` drains `Engine::events` into `input.apply(&event)`, so the window layer
*pushes* events and keeps no second copy of the held/pressed state — the C++ failure
where `WindowInput` and the ImGui layer disagreed about what was down.


### `vault` — assets

```rust
pub struct AssetId { /* index + generation */ }        // Copy, Eq, Hash; generation-checked
pub struct Assets { /* registry: name -> bytes, kind, handle table */ }
impl Assets {
    pub fn new(root: impl Into<PathBuf>) -> Self;
    pub fn mount(&mut self, dir: impl Into<PathBuf>);          // VFS search path
    pub fn load(&mut self, path: &str) -> Result<AssetId>;     // by virtual path
    pub fn load_async(&mut self, jobs: &grind::Jobs, path: &str) -> Result<AssetId>;
    pub fn poll(&mut self) -> Vec<(AssetId, LoadResult)>;
    pub fn bytes(&self, id: AssetId) -> Option<&[u8]>;
    pub fn image(&self, id: AssetId) -> Option<&Image>;        // decoded RGBA8 + extent
    pub fn mesh(&self, id: AssetId) -> Option<&MeshData>;      // CPU-side, no GPU types
    pub fn text(&self, id: AssetId) -> Option<&str>;
    pub fn handle<T>(&self, id: AssetId) -> Option<Handle<T>>;
}
pub struct Image { pub width: u32, pub height: u32, pub rgba8: Vec<u8>, pub srgb: bool }
pub struct MeshData { pub positions: Vec<[f32;3]>, pub normals: Vec<[f32;3]>,
                      pub uvs: Vec<[f32;2]>, pub tangents: Vec<[f32;4]>,
                      pub indices: Vec<u32>, pub material: Option<MaterialData>,
                      pub animations: Vec<AnimationData>, pub skeleton: Option<SkeletonData> }
```

### `meats` — scene

```rust
pub struct Scene { /* hecs World + hierarchy + id map */ }
impl Scene {
    pub fn new() -> Self;
    pub fn spawn(&mut self, name: &str) -> Entity;
    pub fn destroy(&mut self, entity: Entity);
    pub fn get(&self, entity: Entity) -> Option<EntityRef<'_>>;
    pub fn get_mut(&mut self, entity: Entity) -> Option<EntityRefMut<'_>>;
    pub fn entities(&self) -> Vec<Entity>;
    pub fn set_parent(&mut self, child: Entity, parent: Option<Entity>);
    pub fn children(&self, entity: Entity) -> &[Entity];
    pub fn world_transform(&self, entity: Entity) -> glam::Mat4;
    pub fn update_transforms(&mut self);
    pub fn primary_camera(&self) -> Option<Entity>;
    pub fn clear(&mut self);
}
pub fn save(scene: &Scene, path: &Path) -> Result<()>;
pub fn load(path: &Path) -> Result<Scene>;
```

Components (port of `Components.h`, names kept): `Transform`, `Tag`, `Id`, `Camera`,
`MeshRenderer`, `SpriteRenderer`, `Text`, `Light`, `RigidBody`, `Collider`,
`AudioSource`, `Animator`, `SpriteAnimation`, `NativeScript`.

### `adam` — the engine (extended by this port)

```rust
pub struct Engine {
    pub jobs: Arc<grind::Jobs>,
    pub time: Time,
    pub events: pings::EventBus,
    pub input: eyes::Input,
    pub scene: meats::Scene,
    pub assets: vault::Assets,
    pub gpu: Option<Box<dyn chassis::Gpu>>,
    pub physics: Option<Box<dyn PhysicsBackend>>,
    pub audio: Option<Box<dyn AudioBackend>>,
}
pub trait PhysicsBackend: Send { /* step, sync with a &mut meats::Scene */ }
pub trait AudioBackend: Send { /* play/stop/spatial, listener from camera */ }
```

### `bifrost` — renderer frontend

```rust
pub struct Bifrost { /* passes, mesh cache, material cache, cameras, font */ }
impl Bifrost {
    pub fn new(gpu: &mut dyn Gpu, assets: &mut vault::Assets) -> chassis::Result<Self>;
    pub fn resize(&mut self, extent: Extent2d);
    pub fn render(&mut self, gpu: &mut dyn Gpu, jobs: &grind::Jobs, scene: &meats::Scene,
                  view: ViewSource, editor_camera: Option<&EditorCamera>) -> chassis::Result<()>;
}
pub struct BifrostPlugin;   // impl adam::Plugin: build -> Engine owns nothing, plugin owns Bifrost
```

### `wire`, `femur`, `mirror`

Each is `impl adam::Plugin` + a concrete backend. `wire`: `WirePlugin`, `AudioEngine`
(`impl adam::AudioBackend`). `femur`: `FemurPlugin`, `Physics3D`
(`impl adam::PhysicsBackend`). `mirror`: `MirrorPlugin`, `Editor` model + panel state
(`hierarchy`, `inspector`, `content_browser`, `theme`, `gizmo`, `selection`, `play_state`).

### Pinned after the first round

Written down here because they were decided in a conversation and a seam that
only exists in one agent's memory is a seam that gets invented twice.

```rust
// meats — camera
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProjectionType { Perspective, Orthographic }
pub struct SceneCamera { /* fov, near, far, ortho size/near/far, aspect */ }
impl SceneCamera {
    pub fn new_perspective(fov_radians: f32, near: f32, far: f32) -> Self;
    pub fn new_orthographic(size: f32, near: f32, far: f32) -> Self;
    pub fn projection(&self) -> glam::Mat4;
    pub fn projection_type(&self) -> ProjectionType;
    pub fn set_projection_type(&mut self, kind: ProjectionType);
    pub fn set_viewport_size(&mut self, width: f32, height: f32);
    pub fn set_perspective(&mut self, fov_radians: f32, near: f32, far: f32);
    pub fn set_orthographic(&mut self, size: f32, near: f32, far: f32);
    pub fn aspect_ratio(&self) -> f32;
    pub fn fov(&self) -> f32; pub fn near(&self) -> f32; pub fn far(&self) -> f32;
    pub fn orthographic_size(&self) -> f32;
}
pub struct Camera { pub camera: SceneCamera, pub primary: bool, pub fixed_aspect_ratio: bool }
impl Camera { pub fn projection(&self) -> glam::Mat4; }
impl Transform { pub fn matrix(&self) -> glam::Mat4; pub fn view_matrix(&self) -> glam::Mat4; }

// meats — lights and meshes, extracted so the renderer never reads a component
// directly and never guesses where a direction comes from
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LightKind { Directional, Point, Spot }
pub struct Light { pub kind: LightKind, pub color: glam::Vec3, pub intensity: f32,
                   pub radius: f32, pub inner_cone_angle: f32, pub outer_cone_angle: f32,
                   pub enabled: bool }
pub struct LightSample { pub entity: Entity, pub kind: LightKind, pub color: glam::Vec3,
                         pub intensity: f32, pub radius: f32, pub inner_cone_angle: f32,
                         pub outer_cone_angle: f32, pub position: glam::Vec3, pub direction: glam::Vec3 }
pub struct MeshRenderer { pub mesh: Option<vault::AssetId>, pub mesh_path: String,
                          pub material: Option<vault::AssetId>, pub material_path: String,
                          pub tint: glam::Vec4, pub visible: bool, pub cast_shadows: bool }
#[derive(Clone, Copy, Debug)]
pub struct AnimatorState { pub clip: Option<vault::AssetId>, pub time: f32, pub speed: f32,
                           pub looping: bool, pub playing: bool }
pub struct MeshSample { pub entity: Entity, pub mesh: vault::AssetId,
                        pub material: Option<vault::AssetId>, pub tint: glam::Vec4,
                        pub transform: glam::Mat4, pub animator: Option<AnimatorState>,
                        pub visible: bool, pub cast_shadows: bool }
pub struct Text { pub text: String, pub font: Option<vault::AssetId>, pub font_path: String,
                  pub color: glam::Vec4, pub kerning: f32, pub line_spacing: f32 }
impl Scene {
    pub fn lights(&self) -> Vec<LightSample>;   // enabled lights, world transform applied
    pub fn meshes(&self) -> Vec<MeshSample>;    // stable order between calls with no mutation
    /// Fills every `Option<AssetId>` from its `*_path`. Idempotent; a missing
    /// asset is an `AssetError` in the returned list, never a panic.
    pub fn resolve_assets(&mut self, assets: &mut vault::Assets) -> Vec<AssetError>;
    /// The escape hatch a physics or audio backend needs, and the only one.
    pub fn world(&self) -> &hecs::World;
    pub fn world_mut(&mut self) -> &mut hecs::World;
    pub fn id_of(&self, entity: Entity) -> Option<u64>;
    pub fn insert<T: hecs::Component>(&mut self, entity: Entity, component: T) -> bool;
    pub fn remove<T: hecs::Component>(&mut self, entity: Entity) -> Option<T>;
    pub fn has<T: hecs::Component>(&self, entity: Entity) -> bool;
    pub fn update_scripts(&mut self, dt_seconds: f32);
    pub fn set_viewport_size(&mut self, width: u32, height: u32);
    pub fn remove<T: hecs::Component>(&mut self, entity: Entity) -> Option<T>;
}
pub use hecs;                       // a backend queries through `Scene::world`
pub use hecs::{Ref, RefMut};
```

Every component that names an asset pairs the handle with the path it came from
(`mesh`/`mesh_path`, `font`/`font_path`, …): a scene file holds paths, and a
registry-issued `AssetId` cannot be constructed from one.

Units, written down so nobody converts twice: `Transform::rotation` is **Euler
degrees** and `Light::{inner_cone_angle, outer_cone_angle}` are **degrees** (the
C++ and KHR both use radians; `femur` and `bifrost` both assume degrees, which is
the decision). `meats` also ships `Collider`, `ColliderShape`, `RigidBody`,
`RigidBodyKind`, `Script`, `NativeScript`, `Animator`, `SpriteAnimation`,
`AssetError`, and the serializer's error type.


// forge — WGSL, the path the renderer's own shaders take
pub fn to_spirv(source: &str, label: &str) -> Result<Vec<u8>, forge::wgsl::WgslError>;
pub fn entry_points(source: &str, label: &str) -> Result<Vec<(String, forge::Stage)>, forge::wgsl::WgslError>;
pub fn check(source: &str, label: &str) -> Result<(), forge::wgsl::WgslError>;

// stormglass — the windowed path
pub fn create_instance() -> wgpu::Instance;
impl WgpuGpu {
    pub fn with_surface(instance: &wgpu::Instance, surface: wgpu::Surface<'static>,
                        extent: chassis::Extent2d) -> Result<WgpuGpu, WgpuError>;
    pub fn surface(&self) -> Option<&WgpuSurface>;
    pub fn resize(&mut self, extent: chassis::Extent2d) -> Result<(), WgpuError>;
    pub fn present(&mut self) -> Result<(), WgpuError>;
}
impl WgpuSurface {
    pub fn extent(&self) -> chassis::Extent2d;
    pub fn format(&self) -> chassis::TextureFormat;
    pub fn reconfigure(&mut self, extent: chassis::Extent2d) -> Result<bool, WgpuError>;
}
```

`chassis::TextureFormat` also gained `R32G32Float` and `R32G32B32Float`. A vertex
layout needs `vec2`/`vec3` attributes and the enum had no name for them: the wgpu
mapping covers only `Float32`/`Float32x4`, so *no* mesh layout with a position and
a uv was expressible. Vulkan uses one format table for textures and vertex
attributes, so they belong in the existing enum rather than in a second one.

Bootstrapping order, which the windowed binary depends on: a `wgpu` surface can
only be made from a window that exists, and a window only exists inside winit's
event loop, so the GPU exists *after* `App` is constructed. The bin therefore
builds `App` without starting it, and on the first callback does
`instance.create_surface(window.clone())` → `WgpuGpu::with_surface` →
`*app.engine_mut().gpu = Some(..)` → `app.start()` → `app.frame()`, and only
`app.frame()` after that.

## Ported, deliberately

- `UHE/src/UHE/**` — every module outside `Platform/Vulkan`, which the owner has excused.
- `UHEGAME/src/*` — as `UHE-AIMLAB/aimlab`.
- Scene files stay `.unhuman` YAML with the C++ field names, so the existing
  `UHE_EDITOR/assets/scene/*.unhuman` load unchanged.
- The job system, task graph, RHI and render graph are already ported; extend, do not
  rewrite, unless a seam forces it.

## Not ported

| C++ | why |
|---|---|
| `Platform/Windows`, `Platform/Linux` | winit replaces both; the vestigial Linux layer never ran |
| `Renderer2D/Renderer2D.cpp` | excluded from the C++ build (`UHE/CMakeLists.txt:12`) |
| box2d 2D physics | the 3D physics path is the product; 2D comes back when a scene needs it |
| `basis_universal` | needs a C++ toolchain at build time; KTX2 container parsing is ported, UASTC transcode is not |
| msdf-atlas-gen | the *runtime* (atlas JSON + glyph quads) is ported; atlases are pre-baked assets |
| Tracy, spdlog | `log`/`env_logger` for logging; GPU zones via `VulkanScopes`-style debug labels |
