const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "argus_compute",
        .root_source_file = b.path("compute.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Export all pub/export functions
    lib.rdynamic = true;

    b.installArtifact(lib);
}
