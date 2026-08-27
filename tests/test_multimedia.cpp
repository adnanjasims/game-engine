#include "multimedia/audio_pipeline.hpp"
#include "multimedia/video_decoder_stub.hpp"
#include "multimedia/zero_copy_ring.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using eoc::AudioPipeline;
using eoc::RingView;
using eoc::VideoDecoderStub;
using eoc::VideoFrame;
using eoc::ZeroCopyRing;

TEST(ZeroCopyRing, WrapAround) {
  ZeroCopyRing ring(8);
  EXPECT_EQ(ring.capacity(), 8u);
  const std::byte a[] = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5},
                         std::byte{6}};
  EXPECT_EQ(ring.write(a, 6), 6u);
  std::byte head[4];
  EXPECT_EQ(ring.read(head, 4), 4u);
  EXPECT_EQ(static_cast<int>(head[0]), 1);
  const std::byte b[] = {std::byte{7}, std::byte{8}, std::byte{9}, std::byte{10}};
  EXPECT_EQ(ring.write(b, 4), 4u);
  RingView view = ring.acquire_read(6);
  EXPECT_EQ(view.size(), 6u);
  EXPECT_FALSE(view.second.empty());
  ring.commit_read(view.size());
  EXPECT_EQ(ring.size(), 0u);
}

TEST(AudioPipeline, SpscWrapAndFrames) {
  AudioPipeline audio(64);
  std::int16_t in[20];
  for (int i = 0; i < 20; ++i) {
    in[i] = static_cast<std::int16_t>(i);
  }
  EXPECT_EQ(audio.write_s16(in, 20), 20u);
  std::int16_t out[20] = {};
  EXPECT_EQ(audio.read_s16(out, 20), 20u);
  EXPECT_EQ(out[19], 19);
  EXPECT_EQ(audio.size(), 0u);
}

TEST(AudioPipeline, ConcurrentSpsc) {
  AudioPipeline audio(1u << 12);
  constexpr int kCount = 4096;
  std::atomic<int> produced{0};
  std::atomic<int> consumed{0};
  std::thread prod([&]() {
    std::int16_t sample = 1;
    while (produced.load() < kCount) {
      if (audio.write_s16(&sample, 1) == 1) {
        produced.fetch_add(1);
      }
    }
  });
  std::thread cons([&]() {
    std::int16_t sample = 0;
    while (consumed.load() < kCount) {
      if (audio.read_s16(&sample, 1) == 1) {
        EXPECT_EQ(sample, 1);
        consumed.fetch_add(1);
      }
    }
  });
  prod.join();
  cons.join();
  EXPECT_EQ(produced.load(), kCount);
  EXPECT_EQ(consumed.load(), kCount);
}

TEST(VideoDecoder, PatternAndPacket) {
  VideoDecoderStub dec;
  ASSERT_TRUE(dec.open("pattern"));
  EXPECT_EQ(dec.width(), 64u);
  EXPECT_EQ(dec.height(), 48u);
  VideoFrame frame{};
  ASSERT_TRUE(dec.decode_next(frame));
  ASSERT_NE(frame.y, nullptr);
  EXPECT_EQ(frame.width, 64u);
  const auto first = frame.y[0];

  std::vector<std::byte> pkt(dec.frame_bytes(), std::byte{42});
  ASSERT_TRUE(dec.push_packet(pkt.data(), pkt.size(), 99));
  ASSERT_TRUE(dec.decode_next(frame));
  EXPECT_EQ(frame.pts, 99u);
  EXPECT_EQ(frame.y[0], 42);
  EXPECT_NE(first, frame.y[0]);
  EXPECT_GE(dec.frames_decoded(), 2u);
}
