from analysis.particle_distribution import analyze_particle_distribution_x, plot_particle_distribution_x

results = analyze_particle_distribution_x(
    dump_input="/Volumes/Sandisk/post-doc/incline_avalanche/seed_1/results/dump10245000.post",
    num_parts=10,
    exclude_types=[1, 5],
    verbose=True,
)

# Plot with property box included
plot_particle_distribution_x(
    results,
    mode="percentage",
    show_properties_box=True,
    properties_loc="upper right",  # or "side" / "outside"
)
