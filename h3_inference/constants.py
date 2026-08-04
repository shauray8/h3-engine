from __future__ import annotations

from dataclasses import dataclass

FPS = 24
CANVAS_MULTIPLE = 32
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
VAE_SPATIAL_COMPRESSION = 16
# The DiT patch is (t, h, w) = (1, 2, 2), so the video grid is a further /2 in h and w on
# top of the VAE's /16 — an effective /32 per spatial axis.
PATCH_SIZE = (1, 2, 2)

MODALITY_VIDEO, MODALITY_TEXT, MODALITY_AUDIO = 0, 1, 2

def align_num_frames(num_frames: int) -> int:
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames

def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17n + 5, got {num_frames}.")
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2

def audio_latent_num_frames(num_frames: int) -> int:
    """Audio latents per channel covering the video, rounded on the 40 Hz latent grid."""
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))

@dataclass(frozen=True)
class Cell:

    name: str
    width: int
    height: int
    num_frames: int
    steps: int
    num_text_tokens: int = 537
    note: str = ""

    def __post_init__(self) -> None:
        if self.width % CANVAS_MULTIPLE or self.height % CANVAS_MULTIPLE:
            raise ValueError(f"height/width must be multiples of {CANVAS_MULTIPLE}: {self.height}x{self.width}")
        if align_num_frames(self.num_frames) != self.num_frames:
            raise ValueError(f"num_frames must be 17n + 5, got {self.num_frames}")

@dataclass(frozen=True)
class Layout:
    num_latent_frames: int
    latent_height: int
    latent_width: int
    grid_height: int
    grid_width: int
    num_video_rows: int
    num_audio_rows: int
    num_text_rows: int

    @property
    def seq_len(self) -> int:
        return self.num_text_rows + self.num_audio_rows + self.num_video_rows

    @property
    def num_prefix_rows(self) -> int:
        return self.num_text_rows + self.num_audio_rows

def packed_layout(cell: Cell) -> Layout:
    num_latent_frames = video_latent_num_frames(cell.num_frames)
    latent_h = cell.height // VAE_SPATIAL_COMPRESSION
    latent_w = cell.width // VAE_SPATIAL_COMPRESSION
    grid_h = latent_h // PATCH_SIZE[1]
    grid_w = latent_w // PATCH_SIZE[2]
    return Layout(
        num_latent_frames=num_latent_frames,
        latent_height=latent_h,
        latent_width=latent_w,
        grid_height=grid_h,
        grid_width=grid_w,
        num_video_rows=num_latent_frames * grid_h * grid_w,
        num_audio_rows=audio_latent_num_frames(cell.num_frames) * AUDIO_CHANNELS,
        num_text_rows=cell.num_text_tokens,
    )

PRODUCTION = Cell(
    name="production",
    width=1344,
    height=768,
    num_frames=124,
    steps=50,
    note="official t2va cell; 38,247 packed rows;",
)

ACCEPT = Cell(
    name="accept",
    width=768,
    height=448,
    num_frames=124,
    steps=10,
    note="A/B cell; 13,383 packed rows;",
)

PROD_AB = Cell(
    name="prod_ab",
    width=1344,
    height=768,
    num_frames=124,
    steps=6,
    note="production SHAPE (38,247 rows) at gate-grade cost",
)
CELLS = {c.name: c for c in (PRODUCTION, ACCEPT, PROD_AB)}
