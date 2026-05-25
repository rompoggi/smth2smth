4-to-16 frame densify sanity check
==================================
Top row (*_compare.png): duplicate via pick_frame_indices(4, 16) — same as
  dataset.num_frames=16 on 4-frame folders.
Bottom row: 4 uniformly spaced mids per gap (OpenCV flow warp by default).
Green border = original anchor frame.
*_duplicate_playback / *_flow_playback: sequential animations.
