use std::{
    f32::consts::TAU,
    hint::black_box,
    time::Duration,
};

use criterion::{
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
use soundtouch_rs::{
    SpectralPreset,
    SpectralStretch,
};

const SAMPLE_RATE: usize = 44100;
const CHANNELS: usize = 2;
const INPUT_FRAMES: usize = SAMPLE_RATE;
const TEMPO: f64 = 1.5;
const PITCH_SEMI_TONES: f64 = 2.0;
const DRAIN_CHUNK: usize = 4096;

/// Window length of `SpectralPreset::Default`: 0.12 s.
const BLOCK_SAMPLES: usize = SAMPLE_RATE * 12 / 100;
/// Synthesis hop of `SpectralPreset::Default`: 0.03 s.
const INTERVAL_SAMPLES: usize = SAMPLE_RATE * 3 / 100;
/// Input frames consumed per STFT block: `round(interval * tempo)`.
const INPUT_HOP: usize = (INTERVAL_SAMPLES as f64 * TEMPO + 0.5) as usize;
/// Total number of STFT blocks produced for one second of input at the target tempo.
const STFT_BLOCKS: usize = (INPUT_FRAMES - (BLOCK_SAMPLES + INTERVAL_SAMPLES)) / INPUT_HOP + 1;
/// Output frames produced from this input.
const OUTPUT_FRAMES: usize = STFT_BLOCKS * INTERVAL_SAMPLES;
/// Input frames required to produce an identical number of STFT blocks in the reference
/// implementation.
const WRAPPER_INPUT_FRAMES: usize = STFT_BLOCKS * INPUT_HOP;

fn make_input() -> Vec<Vec<f32>> {
    (0..CHANNELS)
        .map(|c| {
            let detune = 0.5f32.mul_add(c as f32, 1.0);
            (0..INPUT_FRAMES)
                .map(|i| {
                    let t = i as f32 / SAMPLE_RATE as f32;
                    let h1 = 0.5 * (TAU * 220.0 * detune * t).sin();
                    let h2 = 0.3 * (TAU * 440.0 * detune * t).sin();
                    let h3 = 0.2 * (TAU * 1320.0 * detune * t).sin();

                    h1 + h2 + h3
                })
                .collect()
        })
        .collect()
}

fn make_interleaved(planar: &[Vec<f32>], frames: usize) -> Vec<f32> {
    let mut interleaved = Vec::with_capacity(frames * CHANNELS);
    for i in 0..frames {
        for channel in planar {
            interleaved.push(channel[i]);
        }
    }
    interleaved
}

fn bench_spectral_pipeline(c: &mut Criterion) {
    let input = make_input();
    let mut output = vec![vec![0.0f32; DRAIN_CHUNK]; CHANNELS];

    let mut stretch = SpectralStretch::builder(CHANNELS, SAMPLE_RATE)
        .preset(SpectralPreset::Default)
        .tempo(TEMPO)
        .pitch_semi_tones(PITCH_SEMI_TONES)
        .build()
        .unwrap();

    let run_once = |stretch: &mut SpectralStretch, output: &mut Vec<Vec<f32>>| -> usize {
        stretch.clear().unwrap();
        stretch.put_samples(&input).unwrap();

        let mut total = 0;
        loop {
            let frames = stretch.receive_samples(output).unwrap();
            if frames == 0 {
                break;
            }
            total += frames;
            for channel in output.iter() {
                black_box(&channel[..frames]);
            }
        }
        total
    };

    let produced = run_once(&mut stretch, &mut output);
    assert_eq!(
        produced, OUTPUT_FRAMES,
        "produced frame count disagrees with the alignment constants"
    );

    let mut group = c.benchmark_group("Spectral Stretch");
    group.throughput(Throughput::Elements((OUTPUT_FRAMES * CHANNELS) as u64));

    group.bench_function(
        "Rust_full_pipeline_44100Hz_stereo_1s_tempo1.5_pitch+2",
        |b| {
            b.iter(|| black_box(run_once(&mut stretch, &mut output)));
        },
    );

    let wrapper_input = make_interleaved(&input, WRAPPER_INPUT_FRAMES);
    let mut wrapper_output = vec![0.0f32; OUTPUT_FRAMES * CHANNELS];
    let mut wrapper = signalsmith_stretch::Stretch::preset_default(
        CHANNELS as u32,
        u32::try_from(SAMPLE_RATE).unwrap(),
    );
    wrapper.set_transpose_factor_semitones(PITCH_SEMI_TONES as f32, None);

    group.bench_function(
        "Wrapper_crate_full_pipeline_44100Hz_stereo_tempo1.5_pitch+2",
        |b| {
            b.iter(|| {
                wrapper.reset();
                wrapper.process(&wrapper_input, &mut wrapper_output);
                black_box(&wrapper_output[..]);
            });
        },
    );

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(5));
    targets = bench_spectral_pipeline
}
criterion_main!(benches);
