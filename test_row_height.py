import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

rows = [
    ["Row 1", "A"],
    ["Row 2", "B"],
    ["", ""],
    ["Row 4", "D"],
]

tbl = ax.table(cellText=rows, loc='center', colWidths=[0.5, 0.5])

# Print cell positions before modification
print("Before modification:")
for (r, c), cell in tbl.get_celld().items():
    print(f"Cell ({r}, {c}): y={cell.get_y():.4f}, h={cell.get_height():.4f}")

# Try setting different height for row 2 (which is 0-indexed data row 2 => table row 2)
spacer_row_idx = 2
for (r, c), cell in tbl.get_celld().items():
    if r == spacer_row_idx:
        cell.set_height(0.02)
    else:
        cell.set_height(0.15)

fig.canvas.draw()
print("\nAfter set_height:")
for (r, c), cell in tbl.get_celld().items():
    print(f"Cell ({r}, {c}): y={cell.get_y():.4f}, h={cell.get_height():.4f}")

plt.savefig("test_row_height.png")
