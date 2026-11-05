import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WIDTH = 460
HEIGHT = 280
RADIUS = 16
BORDER_THICKNESS = 2.5
NUM_FRAMES = 24  # 24 frames for a smooth 1.2-second 360-degree loop (50ms per frame)

# Colors from requirements
COLOR_CYAN = np.array([0, 240, 255], dtype=float)     # #00F0FF
COLOR_PURPLE = np.array([139, 92, 246], dtype=float)  # #8B5CF6
COLOR_GREEN = np.array([0, 255, 136], dtype=float)    # #00FF88
BG_DARK = (6, 11, 20)  # #060B14

font_mono_sm = ImageFont.truetype('consola.ttf', 11)
font_mono_md = ImageFont.truetype('consola.ttf', 13)
font_mono_lg = ImageFont.truetype('consola.ttf', 16)
font_ui_bold = ImageFont.truetype('segoeui.ttf', 12)
font_ui_title = ImageFont.truetype('segoeui.ttf', 14)

def get_gradient_color(angle):
    """Map angle [0, 2pi) to 3-color cyclic gradient: Cyan -> Purple -> Green -> Cyan"""
    a = angle % (2 * math.pi)
    f = a / (2 * math.pi) * 3.0
    idx = int(f)
    t = f - idx
    # smooth cosine interpolation
    t = (1 - math.cos(t * math.pi)) / 2.0
    if idx == 0:
        c = (1 - t) * COLOR_CYAN + t * COLOR_PURPLE
    elif idx == 1:
        c = (1 - t) * COLOR_PURPLE + t * COLOR_GREEN
    else:
        c = (1 - t) * COLOR_GREEN + t * COLOR_CYAN
    return (int(c[0]), int(c[1]), int(c[2]))

def precompute_border_map():
    """Create a mask of border pixels and their corresponding angles from center."""
    cx = WIDTH / 2.0
    cy = HEIGHT / 2.0
    
    # We will sample at 2x resolution for super-smooth antialiasing
    scale = 2
    sw, sh = WIDTH * scale, HEIGHT * scale
    sr = RADIUS * scale
    st = BORDER_THICKNESS * scale
    
    # Create outer rounded rect mask and inner rounded rect mask
    outer = Image.new('L', (sw, sh), 0)
    d_out = ImageDraw.Draw(outer)
    d_out.rounded_rectangle([0, 0, sw - 1, sh - 1], radius=sr, fill=255)
    
    inner = Image.new('L', (sw, sh), 0)
    d_in = ImageDraw.Draw(inner)
    d_in.rounded_rectangle([int(st), int(st), int(sw - 1 - st), int(sh - 1 - st)], radius=int(sr - st), fill=255)
    
    # Border mask
    outer_arr = np.array(outer, dtype=np.uint8)
    inner_arr = np.array(inner, dtype=np.uint8)
    border_mask = (outer_arr > 128) & (inner_arr < 128)
    
    # Calculate angles for each pixel
    y_coords, x_coords = np.nonzero(border_mask)
    dx = (x_coords / scale) - cx
    dy = (y_coords / scale) - cy
    angles = np.arctan2(dy, dx) % (2 * math.pi)
    
    return border_mask, x_coords, y_coords, angles, scale

print("Precomputing border coordinates...")
BORDER_MASK, X_COORDS, Y_COORDS, ANGLES, SCALE = precompute_border_map()
print(f"Precomputed {len(ANGLES)} border subpixels.")

def render_frame_border(rot_offset):
    """Render the rotating gradient border with glow for a specific rotation angle."""
    sw, sh = WIDTH * SCALE, HEIGHT * SCALE
    border_img = Image.new('RGBA', (sw, sh), (0, 0, 0, 0))
    border_arr = np.zeros((sh, sw, 4), dtype=np.uint8)
    
    # Shift angles by rotation offset
    shifted = (ANGLES - rot_offset) % (2 * math.pi)
    
    # Calculate RGB for all border points vectorized
    f = shifted / (2 * math.pi) * 3.0
    idx = f.astype(int)
    t = f - idx
    t = (1.0 - np.cos(t * np.pi)) / 2.0
    
    colors = np.zeros((len(ANGLES), 3), dtype=np.uint8)
    
    m0 = (idx == 0)
    colors[m0] = ((1 - t[m0, None]) * COLOR_CYAN + t[m0, None] * COLOR_PURPLE).astype(np.uint8)
    
    m1 = (idx == 1)
    colors[m1] = ((1 - t[m1, None]) * COLOR_PURPLE + t[m1, None] * COLOR_GREEN).astype(np.uint8)
    
    m2 = (idx >= 2)
    colors[m2] = ((1 - t[m2, None]) * COLOR_GREEN + t[m2, None] * COLOR_CYAN).astype(np.uint8)
    
    border_arr[Y_COORDS, X_COORDS, 0] = colors[:, 0]
    border_arr[Y_COORDS, X_COORDS, 1] = colors[:, 1]
    border_arr[Y_COORDS, X_COORDS, 2] = colors[:, 2]
    border_arr[Y_COORDS, X_COORDS, 3] = 255
    
    # Downsample with high quality antialiasing
    hires = Image.fromarray(border_arr, 'RGBA')
    border_clean = hires.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    
    # Generate subtle neon glow
    glow = border_clean.filter(ImageFilter.GaussianBlur(radius=3.5))
    glow_arr = np.array(glow, dtype=float)
    glow_arr[..., 3] *= 0.6  # subtle glow
    glow = Image.fromarray(glow_arr.astype(np.uint8), 'RGBA')
    
    return border_clean, glow

def draw_card_pms(draw):
    """Draw inner UI content for PMS System"""
    # Top Status Bar
    draw.rectangle([18, 16, 442, 38], fill=(11, 19, 36))
    draw.ellipse([26, 24, 34, 32], fill=(0, 255, 136))
    draw.text((40, 21), "PMS SYSTEM · HOSPITALITY OS", font=font_mono_sm, fill=(248, 250, 252))
    
    # Status pill on right
    draw.rounded_rectangle([335, 19, 434, 35], radius=8, fill=(4, 32, 23), outline=(0, 255, 136), width=1)
    draw.text((345, 21), "v2.4 ONLINE", font=font_mono_sm, fill=(0, 255, 136))
    
    # 3 Metric KPI Cards
    # Tile 1: Occupancy
    draw.rounded_rectangle([18, 48, 150, 110], radius=8, fill=(13, 27, 48), outline=(26, 45, 75), width=1)
    draw.text((26, 56), "OCCUPANCY", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((26, 74), "94.8%", font=font_mono_lg, fill=(0, 240, 255))
    # Sparkline
    pts = [(26, 102), (50, 94), (75, 98), (100, 88), (125, 94), (142, 86)]
    draw.line(pts, fill=(0, 240, 255), width=2)
    
    # Tile 2: Revenue
    draw.rounded_rectangle([160, 48, 298, 110], radius=8, fill=(13, 27, 48), outline=(26, 45, 75), width=1)
    draw.text((168, 56), "REV. TODAY", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((168, 74), "$48,250", font=font_mono_lg, fill=(0, 255, 136))
    # Bar chart
    for i, h in enumerate([14, 22, 18, 28, 24]):
        draw.rectangle([170 + i*24, 104 - h, 184 + i*24, 104], fill=(0, 255, 136))
        
    # Tile 3: Room Status
    draw.rounded_rectangle([308, 48, 442, 110], radius=8, fill=(13, 27, 48), outline=(26, 45, 75), width=1)
    draw.text((316, 56), "ROOM CAPACITY", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((316, 74), "120 / 128", font=font_mono_lg, fill=(250, 204, 21))
    draw.rectangle([316, 96, 432, 102], fill=(20, 36, 60))
    draw.rectangle([316, 96, 424, 102], fill=(250, 204, 21))
    
    # Live Room Booking Grid
    draw.rounded_rectangle([18, 120, 442, 258], radius=8, fill=(8, 15, 28), outline=(20, 36, 60), width=1)
    draw.text((26, 128), "LIVE RESERVATION PIPELINE", font=font_mono_sm, fill=(100, 116, 139))
    
    # Room rows
    rooms = [
        ("301", "Deluxe King", "CHECKED-IN", (4, 32, 23), (0, 255, 136), "Guest: Alex M. (4N)"),
        ("405", "Ocean Suite", "RESERVED", (28, 10, 34), (217, 70, 239), "ETA: 14:30 PM"),
        ("210", "Executive Twin", "CLEANING", (31, 24, 6), (250, 204, 21), "Housekeeping Q3"),
        ("502", "Penthouse VIP", "CHECKED-IN", (4, 32, 23), (0, 255, 136), "Guest: Sarah K. (2N)"),
    ]
    for idx, (num, rtype, status, bg_col, text_col, sub) in enumerate(rooms):
        y = 146 + idx * 26
        draw.text((26, y + 3), f"RM {num}", font=font_mono_sm, fill=(248, 250, 252))
        draw.text((80, y + 3), rtype, font=font_mono_sm, fill=(148, 163, 184))
        draw.rounded_rectangle([210, y, 310, y + 20], radius=4, fill=bg_col, outline=text_col, width=1)
        draw.text((220, y + 3), status, font=font_mono_sm, fill=text_col)
        draw.text((320, y + 3), sub, font=font_mono_sm, fill=(100, 116, 139))

def draw_card_moderna(draw):
    """Draw inner UI content for MODERNA Clothing Store"""
    # Top Store Navbar
    draw.rectangle([18, 16, 442, 38], fill=(11, 19, 36))
    draw.text((28, 21), "✦ MODERNA", font=font_mono_md, fill=(248, 250, 252))
    draw.text((125, 22), "STREETWEAR · RUNWAY · DROP 04", font=font_mono_sm, fill=(139, 92, 246))
    
    draw.rounded_rectangle([356, 19, 434, 35], radius=8, fill=(28, 10, 34), outline=(217, 70, 239), width=1)
    draw.text((368, 21), "CART [2]", font=font_mono_sm, fill=(217, 70, 239))
    
    # Store Hero Section Left (Product Spotlight)
    draw.rounded_rectangle([18, 48, 250, 258], radius=8, fill=(10, 16, 30), outline=(26, 45, 75), width=1)
    
    # Clothing item graphic placeholder
    draw.rounded_rectangle([32, 60, 236, 165], radius=6, fill=(16, 26, 46))
    # Draw stylized cyber bomber jacket silhouette
    draw.polygon([(134, 75), (100, 95), (108, 155), (160, 155), (168, 95)], fill=(139, 92, 246))
    draw.polygon([(134, 75), (115, 88), (115, 140), (134, 150)], fill=(0, 240, 255))
    draw.line([(134, 75), (134, 155)], fill=(255, 255, 255), width=2)
    # Neon badge on image
    draw.rounded_rectangle([40, 68, 95, 84], radius=4, fill=(4, 32, 23), outline=(0, 255, 136), width=1)
    draw.text((46, 70), "NEW DROP", font=font_mono_sm, fill=(0, 255, 136))
    
    # Product Meta
    draw.text((32, 175), "Cyber Oversized Bomber", font=font_ui_bold, fill=(255, 255, 255))
    draw.text((32, 194), "$189.00 USD", font=font_mono_md, fill=(0, 240, 255))
    draw.text((140, 195), "In Stock (M / L / XL)", font=font_mono_sm, fill=(100, 116, 139))
    
    # Add to cart button
    draw.rounded_rectangle([32, 218, 236, 248], radius=6, fill=(139, 92, 246))
    draw.text((82, 224), "ADD TO BAG 🛍️", font=font_mono_sm, fill=(255, 255, 255))
    
    # Store Grid Right (Categories & Trending Items)
    draw.rounded_rectangle([260, 48, 442, 148], radius=8, fill=(10, 16, 30), outline=(26, 45, 75), width=1)
    draw.text((270, 56), "TRENDING COLLECTIONS", font=font_mono_sm, fill=(100, 116, 139))
    
    cats = [
        ("Techwear Cargos", "$120", (0, 240, 255)),
        ("Hologram Hoodie", "$95", (0, 255, 136)),
        ("Tactical Vest", "$145", (250, 204, 21))
    ]
    for idx, (cname, price, col) in enumerate(cats):
        cy = 78 + idx * 22
        draw.ellipse([272, cy + 4, 278, cy + 10], fill=col)
        draw.text((286, cy), cname, font=font_mono_sm, fill=(248, 250, 252))
        draw.text((400, cy), price, font=font_mono_sm, fill=col)
        
    # Bottom checkout security / stack info
    draw.rounded_rectangle([260, 158, 442, 258], radius=8, fill=(8, 15, 28), outline=(20, 36, 60), width=1)
    draw.text((270, 168), "STORE ARCHITECTURE", font=font_mono_sm, fill=(100, 116, 139))
    draw.text((270, 190), "● Node.js / Express Backend", font=font_mono_sm, fill=(0, 255, 136))
    draw.text((270, 210), "● MongoDB Product Catalog", font=font_mono_sm, fill=(0, 240, 255))
    draw.text((270, 230), "● Responsive Cart & Checkout", font=font_mono_sm, fill=(139, 92, 246))

def draw_card_pizza(draw):
    """Draw inner UI content for Pizza Management System"""
    # Top Odoo POS Bar
    draw.rectangle([18, 16, 442, 38], fill=(11, 19, 36))
    draw.ellipse([26, 24, 34, 32], fill=(250, 204, 21))
    draw.text((40, 21), "PIZZA POS & KDS · ODOO 18 ERP", font=font_mono_sm, fill=(248, 250, 252))
    
    draw.rounded_rectangle([326, 19, 434, 35], radius=8, fill=(19, 14, 31), outline=(139, 92, 246), width=1)
    draw.text((336, 21), "PYTHON ODOO", font=font_mono_sm, fill=(139, 92, 246))
    
    # Kitchen Display System (KDS) Kanban Columns
    cols = [
        ("INCOMING (2)", 18, 154, (28, 10, 34), (217, 70, 239)),
        ("IN OVEN (3)", 162, 298, (31, 24, 6), (250, 204, 21)),
        ("READY (4)", 306, 442, (4, 32, 23), (0, 255, 136)),
    ]
    for title, x1, x2, bg_col, text_col in cols:
        draw.rounded_rectangle([x1, 48, x2, 74], radius=6, fill=bg_col, outline=text_col, width=1)
        draw.text((x1 + 10, 54), title, font=font_mono_sm, fill=text_col)
        
    # Kanban Cards in Columns
    # Card 1 (Incoming):
    draw.rounded_rectangle([18, 82, 154, 160], radius=6, fill=(10, 18, 32), outline=(26, 45, 75), width=1)
    draw.text((24, 88), "ORDER #1084", font=font_mono_sm, fill=(248, 250, 252))
    draw.text((24, 104), "1x Truffle Beef L", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((24, 120), "1x Garlic Bread", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((24, 140), "⏳ 02:45 min", font=font_mono_sm, fill=(217, 70, 239))
    
    draw.rounded_rectangle([18, 168, 154, 246], radius=6, fill=(10, 18, 32), outline=(26, 45, 75), width=1)
    draw.text((24, 174), "ORDER #1085", font=font_mono_sm, fill=(248, 250, 252))
    draw.text((24, 190), "2x Pepperoni Hot", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((24, 206), "1x Coke Zero", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((24, 226), "⏳ 00:30 min", font=font_mono_sm, fill=(217, 70, 239))
    
    # Column 2 (In Oven):
    draw.rounded_rectangle([162, 82, 298, 160], radius=6, fill=(10, 18, 32), outline=(26, 45, 75), width=1)
    draw.text((168, 88), "ORDER #1081", font=font_mono_sm, fill=(248, 250, 252))
    draw.text((168, 104), "1x 4-Cheese Melt", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((168, 120), "Oven Deck #2", font=font_mono_sm, fill=(250, 204, 21))
    draw.text((168, 140), "🔥 04:12 / 07:00", font=font_mono_sm, fill=(250, 204, 21))
    
    draw.rounded_rectangle([162, 168, 298, 246], radius=6, fill=(10, 18, 32), outline=(26, 45, 75), width=1)
    draw.text((168, 174), "ORDER #1082", font=font_mono_sm, fill=(248, 250, 252))
    draw.text((168, 190), "1x BBQ Chicken L", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((168, 206), "Oven Deck #1", font=font_mono_sm, fill=(250, 204, 21))
    draw.text((168, 226), "🔥 06:10 / 07:00", font=font_mono_sm, fill=(250, 204, 21))
    
    # Column 3 (Ready for Delivery):
    draw.rounded_rectangle([306, 82, 442, 160], radius=6, fill=(10, 18, 32), outline=(26, 45, 75), width=1)
    draw.text((312, 88), "ORDER #1078", font=font_mono_sm, fill=(248, 250, 252))
    draw.text((312, 104), "1x Seafood Pesto", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((312, 120), "Rider: T. Nam", font=font_mono_sm, fill=(0, 255, 136))
    draw.text((312, 140), "🚀 DISPATCHED", font=font_mono_sm, fill=(0, 255, 136))
    
    draw.rounded_rectangle([306, 168, 442, 246], radius=6, fill=(10, 18, 32), outline=(26, 45, 75), width=1)
    draw.text((312, 174), "ORDER #1079", font=font_mono_sm, fill=(248, 250, 252))
    draw.text((312, 190), "Takeaway Table 4", font=font_mono_sm, fill=(148, 163, 184))
    draw.text((312, 206), "Paid POS #1", font=font_mono_sm, fill=(0, 240, 255))
    draw.text((312, 226), "✅ COMPLETED", font=font_mono_sm, fill=(0, 255, 136))

def generate_project_gif(name, draw_content_fn, output_path):
    print(f"Generating {output_path}...")
    frames = []
    
    # Base inner card background
    inner_base = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    d_inner = ImageDraw.Draw(inner_base)
    d_inner.rounded_rectangle([2, 2, WIDTH - 3, HEIGHT - 3], radius=RADIUS, fill=BG_DARK)
    draw_content_fn(d_inner)
    
    for i in range(NUM_FRAMES):
        angle = (2 * math.pi * i) / NUM_FRAMES
        border_clean, glow = render_frame_border(angle)
        
        # Composite layers: glow -> inner card -> border
        frame = Image.new('RGBA', (WIDTH, HEIGHT), (3, 5, 10, 255))
        frame.alpha_composite(glow)
        frame.alpha_composite(inner_base)
        frame.alpha_composite(border_clean)
        
        # Convert to RGB with optimal palette for crisp GIF output
        rgb_frame = frame.convert('RGB')
        p_frame = rgb_frame.convert('P', palette=Image.Palette.ADAPTIVE, colors=96)
        frames.append(p_frame)
        
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,  # 50ms per frame = 20 fps, 1.2s seamless loop
        loop=0,
        optimize=True
    )
    size_kb = os.path.getsize(output_path) / 1024.0
    print(f"Successfully generated {output_path} ({size_kb:.1f} KB)")

os.makedirs('assets/projects', exist_ok=True)
generate_project_gif("pms-system", draw_card_pms, "assets/projects/pms-system.gif")
generate_project_gif("moderna-store", draw_card_moderna, "assets/projects/moderna-store.gif")
generate_project_gif("pizza-management", draw_card_pizza, "assets/projects/pizza-management.gif")
print("ALL GIFS GENERATED SUCCESSFULLY!")

