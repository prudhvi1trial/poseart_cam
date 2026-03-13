import customtkinter as ctk  # type: ignore
import cv2  # type: ignore
import time
import math
import random
import os
from PIL import Image  # type: ignore
from pose_detector import PoseDetector
from holistic_detector import HolisticDetector
import numpy as np  # type: ignore

# Set global appearance mode and color theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pose Art Studio - Professional Edition")
        self.geometry("1200x700")
        self.minsize(900, 600)

        # Main grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ----------------- SIDEBAR -----------------
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(2, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Pose Art Studio", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Tabview for Controls & Settings
        self.tabview = ctk.CTkTabview(self.sidebar_frame, width=230)
        self.tabview.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.tabview.add("Controls")
        self.tabview.add("Appearance")

        # --- Controls Tab ---
        self.tabview.tab("Controls").grid_columnconfigure(0, weight=1)
        
        self.btn_toggle_cam = ctk.CTkButton(self.tabview.tab("Controls"), text="Stop Camera", command=self.toggle_camera, height=40, font=ctk.CTkFont(weight="bold"))
        self.btn_toggle_cam.grid(row=0, column=0, padx=20, pady=(20, 20), sticky="ew")

        self.switch_draw_original = ctk.CTkSwitch(self.tabview.tab("Controls"), text="Overlay Skeleton (Camera)")
        self.switch_draw_original.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="w")
        
        self.switch_draw_stickman = ctk.CTkSwitch(self.tabview.tab("Controls"), text="Enable Art Feed")
        self.switch_draw_stickman.select()  # enabled by default
        self.switch_draw_stickman.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="w")

        self.switch_gesture_control = ctk.CTkSwitch(self.tabview.tab("Controls"), text="Enable Gesture Control")
        # self.switch_gesture_control.select()  # disabled by default per user request
        self.switch_gesture_control.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")

        # --- Appearance Tab ---
        self.tabview.tab("Appearance").grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.tabview.tab("Appearance"), text="Color Theme:", anchor="w").grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")
        self.color_themes = {
            "Neon Glow (Yellow/Green)": {"outline": (0, 255, 255), "inner": (50, 255, 100), "joint": (255, 100, 100)},
            "Cyberpunk (Pink/Purple)": {"outline": (255, 0, 255), "inner": (100, 50, 255), "joint": (100, 255, 100)},
            "Digital Ice (Cyan/Blue)": {"outline": (255, 255, 0), "inner": (255, 50, 50), "joint": (100, 100, 255)},
            "Monochrome (White/Silver)": {"outline": (255, 255, 255), "inner": (200, 200, 200), "joint": (50, 50, 50)}
        }
        self.theme_menu = ctk.CTkOptionMenu(self.tabview.tab("Appearance"), values=list(self.color_themes.keys()))
        self.theme_menu.grid(row=1, column=0, padx=20, pady=(5, 15), sticky="ew")

        ctk.CTkLabel(self.tabview.tab("Appearance"), text="Canvas Background:", anchor="w").grid(row=2, column=0, padx=20, pady=(5, 0), sticky="w")
        self.bg_menu = ctk.CTkOptionMenu(self.tabview.tab("Appearance"), values=["Solid Black", "Transparent (Camera)"])
        self.bg_menu.grid(row=3, column=0, padx=20, pady=(5, 15), sticky="ew")

        ctk.CTkLabel(self.tabview.tab("Appearance"), text="Line Thickness:", anchor="w").grid(row=4, column=0, padx=20, pady=(5, 0), sticky="w")
        self.slider_thickness = ctk.CTkSlider(self.tabview.tab("Appearance"), from_=2, to=25, number_of_steps=23)
        self.slider_thickness.set(10)
        self.slider_thickness.grid(row=5, column=0, padx=20, pady=(5, 15), sticky="ew")

        ctk.CTkLabel(self.tabview.tab("Appearance"), text="Joint Node Size:", anchor="w").grid(row=6, column=0, padx=20, pady=(5, 0), sticky="w")
        self.slider_nodes = ctk.CTkSlider(self.tabview.tab("Appearance"), from_=0, to=30, number_of_steps=30)
        self.slider_nodes.set(12)
        self.slider_nodes.grid(row=7, column=0, padx=20, pady=(5, 15), sticky="ew")

        ctk.CTkLabel(self.tabview.tab("Appearance"), text="Rendering Style:", anchor="w").grid(row=8, column=0, padx=20, pady=(5, 0), sticky="w")
        self.style_menu = ctk.CTkOptionMenu(self.tabview.tab("Appearance"), values=["Bubble Man", "Hell Fire", "Shadow Void", "Magic Button", "Classic Stickman", "Minimalist Line Art", "Anatomical Skeleton", "Ultimate 3D Wireframe"])
        self.style_menu.set("Classic Stickman")
        self.style_menu.grid(row=9, column=0, padx=20, pady=(5, 15), sticky="ew")


        # Footer info in sidebar
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Ready", text_color="gray")
        self.status_label.grid(row=3, column=0, padx=20, pady=20, sticky="sw")

        # ----------------- MAIN VIDEO PANEL -----------------
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        # Configure layout inside video_frame
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(1, weight=1)
        self.video_frame.grid_rowconfigure(1, weight=1)

        # Titles for the feeds
        ctk.CTkLabel(self.video_frame, text="Live Camera Feed", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=(10, 0))
        ctk.CTkLabel(self.video_frame, text="Pose Art Canvas", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=1, pady=(10, 0))

        # Image placeholders
        self.lbl_video = ctk.CTkLabel(self.video_frame, text="")
        self.lbl_video.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.lbl_art = ctk.CTkLabel(self.video_frame, text="")
        self.lbl_art.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Blank placeholder image for clearing panels cleanly
        self._blank_img = ctk.CTkImage(light_image=Image.new("RGB", (1, 1), (0, 0, 0)), size=(1, 1))

        # ----------------- APPLICATION LOGIC -----------------
        self.cap = cv2.VideoCapture(0)
        self.detector = PoseDetector(min_pose_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.holistic_detector = HolisticDetector(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.camera_running = True
        self.particles = []  # List to store smoke particles
        
        # Load custom background for Shadow Void
        self.bg_image_path = os.path.join("assets", "room_bg.jpg")
        self.bg_image_raw = cv2.imread(self.bg_image_path) if os.path.exists(self.bg_image_path) else None
        
        # Magic Button Assets
        self.btn_image_raw = cv2.imread(os.path.join("assets", "red_button.png"), cv2.IMREAD_UNCHANGED)
        self.sparkle_image_raw = cv2.imread(os.path.join("assets", "blue_sparkle.png"), cv2.IMREAD_UNCHANGED)
        self.magic_button_active = False
        self.button_rect = (0, 0, 0, 0) # Placeholder
        self.button_pos = None # (x, y) coordinates of button center
        self.magic_touch_count = 0 
        self.magic_bg_color = (180, 105, 255) # Initial Hot Pink
        self.magic_stickman_theme = {"outline": (255, 200, 0), "inner": (255, 255, 100), "joint": (255, 150, 0)}
        self.magic_cooldown = 0
        self.frame_count = 0 # For frame-skipping logic
        self.holistic_face_results = None
        self.holistic_hand_results = None
        
        # --- Gesture Control State ---
        self.gesture_history = []  # List of (x, y) for right wrist
        self.gesture_cooldown = 0   # Frames until next gesture allowed
        self.swipe_threshold = 100 # Minimum horizontal movement
        self.history_len = 15      # Frames to track for gesture
        
        # --- Auto-Styling State ---
        self.last_style_change_time = 0.0
        
        # --- Minimalist Line Art State ---
        self.miniline_canvas = None  # Persistent trail canvas
        # Background subtractor to separate person from bg
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

        self.update_video()

    def toggle_camera(self):
        if self.camera_running:
            self.camera_running = False
            self.btn_toggle_cam.configure(text="Start Camera", fg_color="green", hover_color="darkgreen")
            self.status_label.configure(text="Status: Paused")
            if self.cap.isOpened():
                self.cap.release()
            self.lbl_video.configure(image=self._blank_img)
            self.lbl_art.configure(image=self._blank_img)
        else:
            self.camera_running = True
            self.btn_toggle_cam.configure(text="Stop Camera", fg_color=["#3B8ED0", "#1F6AA5"], hover_color=["#36719F", "#144870"])
            self.status_label.configure(text="Status: Running")
            self.cap = cv2.VideoCapture(0)
            self.update_video()

    def draw_stickman(self, img, lm_list):
        h, w, c = img.shape
        
        # Canvas Mode Handling
        style = self.style_menu.get()
        if style == "Minimalist Line Art":
            # White canvas with ghosting trail effect
            if self.miniline_canvas is None or self.miniline_canvas.shape[:2] != (h, w):
                self.miniline_canvas = np.ones((h, w, c), dtype=np.uint8) * 255
            # Fade previous canvas toward white to create trail effect
            white = np.ones((h, w, c), dtype=np.uint8) * 255
            self.miniline_canvas = cv2.addWeighted(self.miniline_canvas, 0.75, white, 0.25, 0)
            art_img = self.miniline_canvas.copy()
        elif style == "Anatomical Skeleton":
            art_img = np.zeros((h, w, c), dtype=np.uint8)
        elif style in ["Shadow Void", "Magic Button"] and self.bg_image_raw is not None:
            art_img = cv2.resize(self.bg_image_raw, (w, h))
            if style == "Magic Button" and self.magic_button_active:
                # Dynamic Random Background
                overlay = np.full(art_img.shape, self.magic_bg_color, dtype=np.uint8)
                art_img = cv2.addWeighted(art_img, 0.5, overlay, 0.5, 0)
        elif style == "Ultimate 3D Wireframe":
            art_img = np.zeros((h, w, c), dtype=np.uint8)
        elif self.bg_menu.get() == "Solid Black":
            art_img = np.zeros((h, w, c), dtype=np.uint8)
        else:
            # Darkened camera background to make art pop
            art_img = cv2.convertScaleAbs(img, alpha=0.3, beta=0) 
        
        if len(lm_list) != 0:
            connections = [
                (11, 12), # shoulders
                (11, 13), (13, 15), # left arm
                (12, 14), (14, 16), # right arm
                (11, 23), (12, 24), (23, 24), # torso
                (23, 25), (25, 27), # left leg
                (24, 26), (26, 28), # right leg
            ]
            
            
            points = {lm[0]: (lm[1], lm[2]) for lm in lm_list}
            
            # Get current dynamic settings
            style = self.style_menu.get()
            theme = self.color_themes[self.theme_menu.get()]
            
            # --- Hell Fire / Shadow Void Theme Prep ---
            if style == "Hell Fire":
                hell_theme = {"outline": (0, 0, 255), "inner": (100, 100, 255), "joint": (0, 0, 255)}
                theme = hell_theme
            elif style == "Shadow Void":
                void_theme = {"outline": (30, 30, 30), "inner": (0, 0, 0), "joint": (30, 30, 30)}
                theme = void_theme
            elif style == "Magic Button":
                if self.magic_button_active:
                    # Dynamic Random Stickman Theme
                    theme = self.magic_stickman_theme
                else:
                    # Shadow Void Base (Black)
                    theme = {"outline": (30, 30, 30), "inner": (0, 0, 0), "joint": (30, 30, 30)}
            elif style == "Ultimate 3D Wireframe":
                # Clean White mesh
                theme = {"outline": (255, 255, 255), "inner": (200, 200, 200), "joint": (255, 255, 255)}
            elif style == "Anatomical Skeleton":
                # Off-white / Bone color
                theme = {"outline": (220, 230, 235), "inner": (180, 190, 200), "joint": (220, 230, 235)}
                
            thickness = int(self.slider_thickness.get())
            if style == "Anatomical Skeleton":
                thickness = max(2, min(6, thickness // 3)) # Slimmer digital/realistic look
            inner_thickness = max(1, int(thickness / 2)) # Inner core is half size
            node_sz = int(self.slider_nodes.get())
            
            # Draw Head (Skip for 3D Wireframe to show only landmarked face)
            if 0 in points and style != "Ultimate 3D Wireframe":
                neck_y = points[0][1] + 30 
                if 11 in points and 12 in points:
                    neck_y = int((points[11][1] + points[12][1]) / 2) - 10
                
                cv2.line(art_img, points[0], (points[0][0], neck_y), theme["outline"], thickness)
                cv2.line(art_img, points[0], (points[0][0], neck_y), theme["inner"], inner_thickness)
                
                head_radius = max(10, int(thickness * 2.5))
                if style == "Anatomical Skeleton":
                    head_radius = max(20, int(thickness * 4.5)) # Larger skull
                    # Detailed Skull shape
                    cv2.ellipse(art_img, points[0], (int(head_radius), int(head_radius*1.2)), 0, 0, 360, theme["outline"], cv2.FILLED)
                    # Jaw line
                    jaw_y = points[0][1] + int(head_radius * 1.2)
                    cv2.rectangle(art_img, (points[0][0]-int(head_radius*0.6), points[0][1]), (points[0][0]+int(head_radius*0.6), jaw_y), theme["outline"], cv2.FILLED)
                    # Eye sockets
                    eye_off = int(head_radius * 0.4)
                    cv2.circle(art_img, (points[0][0] - eye_off, points[0][1]), int(head_radius*0.3), (20, 20, 20), -1)
                    cv2.circle(art_img, (points[0][0] + eye_off, points[0][1]), int(head_radius*0.3), (20, 20, 20), -1)
                    # Nose
                    cv2.circle(art_img, (points[0][0], points[0][1] + int(eye_off*0.8)), int(head_radius*0.15), (20, 20, 20), -1)
                    
                    # --- Skull Fire Emitter ---
                    if random.random() > 0.2:
                        for _ in range(3):
                            self.particles.append({
                                "x": points[0][0] + random.uniform(-head_radius*0.5, head_radius*0.5),
                                "y": neck_y + random.uniform(-5, 5),
                                "vx": random.uniform(-1.5, 1.5),
                                "vy": random.uniform(-8, -3), # Rise fast like fire
                                "life": 1.0,
                                "fade_speed": random.uniform(0.04, 0.08),
                                "size": random.uniform(10, 25)
                            })
                else:
                    cv2.circle(art_img, points[0], int(head_radius), theme["outline"], cv2.FILLED)
                    cv2.circle(art_img, points[0], int(head_radius + 5), (255, 255, 255), max(1, int(inner_thickness/2)))
                
            # Draw Torso
            if style == "Anatomical Skeleton":
                if all(k in points for k in [11, 12, 23, 24]):
                    # Spine connection
                    spine_top = ((points[11][0] + points[12][0]) // 2, (points[11][1] + points[12][1]) // 2)
                    spine_bottom = ((points[23][0] + points[24][0]) // 2, (points[23][1] + points[24][1]) // 2)
                    cv2.line(art_img, spine_top, spine_bottom, theme["outline"], thickness + 2)
                    
                    # Ribcage
                    for i in range(1, 6):
                        t = i / 6.0
                        ry = int(spine_top[1] + (spine_bottom[1] - spine_top[1]) * t)
                        rx = int(spine_top[0] + (spine_bottom[0] - spine_top[0]) * t)
                        # Rib width decreases slightly down
                        rw = int((abs(points[11][0] - points[12][0])) * (0.8 - t*0.2))
                        cv2.ellipse(art_img, (rx, ry), (rw // 2, thickness * 2), 0, 0, 360, theme["outline"], thickness)
                    
                    # Pelvis structure
                    p_pts = np.array([
                        [points[23][0] - 10, points[23][1] - 10],
                        [points[24][0] + 10, points[24][1] - 10],
                        [spine_bottom[0], spine_bottom[1] + 20]
                    ], np.int32)
                    cv2.fillPoly(art_img, [p_pts], theme["outline"])
                    cv2.circle(art_img, (points[23][0], points[23][1]), thickness * 2, theme["outline"], -1)
                    cv2.circle(art_img, (points[24][0], points[24][1]), thickness * 2, theme["outline"], -1)

            elif all(k in points for k in [11, 12, 23, 24]):
                pts = np.array([points[11], points[12], points[24], points[23]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                overlay = art_img.copy()
                cv2.fillPoly(overlay, [pts], theme["inner"])
                alpha = 0.5
                art_img = cv2.addWeighted(overlay, alpha, art_img, 1 - alpha, 0)
            # Draw Lines or Bubbles
            
            # --- Hell Fire / Shadow Void / Anatomical Skeleton Special Prep ---
            if style in ["Hell Fire", "Shadow Void", "Anatomical Skeleton"]:
                if style == "Hell Fire":
                    hell_theme = {"outline": (0, 0, 255), "inner": (100, 100, 255), "joint": (0, 0, 255)}
                    theme = hell_theme
                elif style == "Anatomical Skeleton":
                     theme = {"outline": (220, 230, 235), "inner": (180, 190, 200), "joint": (220, 230, 235)}
                else: # Shadow Void
                    void_theme = {"outline": (30, 30, 30), "inner": (0, 0, 0), "joint": (30, 30, 30)}
                    theme = void_theme
                
                radius = max(10, int(thickness * 2.5))
                spacing = radius * 0.8
                
                # Update and Draw Particles
                new_particles = []
                overlay = art_img.copy()
                for p in self.particles:
                    p["life"] -= p["fade_speed"]
                    if p["life"] > 0:
                        p["x"] += p["vx"]
                        p["y"] += p["vy"]
                        p["vy"] -= 0.6 if style == "Hell Fire" else 0.3 # Void floats slower
                        p["vx"] += random.uniform(-1, 1)
                        p["size"] *= 1.05
                        
                        sz = int(p["size"] * p["life"])
                        if sz > 0:
                            # Draw smoke/sparkle particle
                            if style == "Hell Fire":
                                color = (0, 0, 200)
                            elif style == "Magic Button":
                                if self.magic_button_active:
                                    color = (255, 255, 150) # Bright cyan-white for sparks
                                    sz = int(sz * 0.8) # Smaller sparks
                                else:
                                    color = (20, 20, 20)
                            elif style == "Anatomical Skeleton":
                                # Fire colors: Red, Orange, Yellow in BGR
                                color = random.choice([(20, 20, 220), (20, 100, 255), (20, 200, 255)])
                            else: # Shadow Void
                                color = (20, 20, 20)
                                
                            cv2.circle(overlay, (int(p["x"]), int(p["y"])), min(200, sz), color, -1)
                        new_particles.append(p)
                self.particles = new_particles
                alpha = 0.4 if style == "Hell Fire" else 0.6
                art_img = cv2.addWeighted(overlay, alpha, art_img, 1 - alpha, 0)

            # --- High-Level 3D Wireframe Rendering (Body) ---
            if style == "Ultimate 3D Wireframe" and (self.holistic_face_results or self.holistic_hand_results):
                # 1. Draw Body Volumetric Rings (Optimized)
                for connection in connections:
                    if connection[0] in points and connection[1] in points:
                        p1, p2 = points[connection[0]], points[connection[1]]
                        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        # Calculate fewer volumetric rings (every 40px)
                        steps = max(2, int(dist / 40))
                        for i in range(steps + 1):
                            t = i / steps
                            cx = int(p1[0] + (p2[0] - p1[0]) * t)
                            cy = int(p1[1] + (p2[1] - p1[1]) * t)
                            
                            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) + math.pi/2
                            r = int(thickness * 1.2)
                            if connection in [(11, 23), (12, 24), (23, 24), (11, 12)]: # Torso
                                r = int(thickness * 2.8)
                                
                            # Simpler rings (6 points instead of 8)
                            pts = []
                            for deg in range(0, 360, 60):
                                rad = math.radians(deg)
                                rx = cx + int(r * math.cos(rad) * math.cos(angle))
                                ry = cy + int(r * math.cos(rad) * math.sin(angle))
                                rz = int(r * 0.4 * math.sin(rad)) # Shallower depth
                                final_x = rx + int(rz * math.cos(angle - math.pi/2))
                                final_y = ry + int(rz * math.sin(angle - math.pi/2))
                                pts.append([final_x, final_y])
                            
                            cv2.polylines(art_img, [np.array(pts, np.int32)], True, theme["outline"], 1)

                # 2. Draw Face and Hands via HolisticDetector
                self.holistic_detector.draw_ultimate_art(art_img, self.holistic_face_results, self.holistic_hand_results, theme["outline"])
                return art_img # Early exit for 3D style

            for connection in connections:
                if connection[0] in points and connection[1] in points:
                    p1 = points[connection[0]]
                    p2 = points[connection[1]]
                    
                    if style == "Classic Stickman":
                        cv2.line(art_img, p1, p2, theme["outline"], thickness)
                        cv2.line(art_img, p1, p2, theme["inner"], inner_thickness)
                    elif style == "Anatomical Skeleton":
                        # Double bones for forearm and lower leg
                        if connection in [(13, 15), (14, 16), (25, 27), (26, 28)]:
                            # Offset lines for Radius/Ulna or Tibia/Fibula
                            off = 4
                            cv2.line(art_img, (p1[0]-off, p1[1]), (p2[0]-off, p2[1]), theme["outline"], max(1, thickness-1))
                            cv2.line(art_img, (p1[0]+off, p1[1]), (p2[0]+off, p2[1]), theme["outline"], max(1, thickness-1))
                        else:
                            cv2.line(art_img, p1, p2, theme["outline"], thickness + 1)
                    elif style == "Bubble Man":
                        # Calculate distance and steps
                        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        
                        # Match bubble size to head size
                        radius = max(10, int(thickness * 2.5))
                        
                        # Set spacing proportional to bubble size (approx 1.5x radius)
                        spacing = radius * 1.5
                        steps = max(2, int(dist / spacing))
                        
                        for i in range(steps + 1):
                            t = i / steps
                            cx = int(p1[0] + (p2[0] - p1[0]) * t)
                            cy = int(p1[1] + (p2[1] - p1[1]) * t)
                            
                            # Draw bubble with outline and inner highlight
                            cv2.circle(art_img, (cx, cy), radius, theme["outline"], cv2.FILLED)
                            cv2.circle(art_img, (cx, cy), max(1, int(radius * 0.6)), theme["inner"], cv2.FILLED)
                            # Tiny white specular highlight
                            if radius > 4:
                                cv2.circle(art_img, (cx - int(radius*0.3), cy - int(radius*0.3)), max(1, int(radius*0.15)), (255, 255, 255), cv2.FILLED)
                    elif style in ["Hell Fire", "Shadow Void"]:
                        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        steps = max(2, int(dist / spacing))
                        
                        for i in range(steps + 1):
                            t = i / steps
                            cx = int(p1[0] + (p2[0] - p1[0]) * t)
                            cy = int(p1[1] + (p2[1] - p1[1]) * t)
                            
                            # Spawn smoke particles
                            if random.random() > 0.7:
                                self.particles.append({
                                    "x": cx + random.uniform(-5, 5),
                                    "y": cy + random.uniform(-5, 5),
                                    "vx": random.uniform(-2, 2),
                                    "vy": random.uniform(-2, 1) if style == "Magic Button" else random.uniform(-2, 0),
                                    "life": 1.0,
                                    "fade_speed": 0.04 if style == "Hell Fire" else 0.03,
                                    "size": random.uniform(head_radius*0.8, head_radius*1.2) if style == "Magic Button" else random.uniform(radius*0.5, radius*1.5)
                                })
                            
                            # Draw Core
                            if style == "Hell Fire":
                                color_outer = (0, 0, 255)
                                color_inner = (200, 200, 255)
                            elif style == "Magic Button":
                                color_outer = (255, 50, 0) if self.magic_button_active else (30, 30, 30)
                                color_inner = (255, 200, 100) if self.magic_button_active else (0, 0, 0)
                            else: # Shadow Void
                                color_outer = (30, 30, 30)
                                color_inner = (0, 0, 0)
                            
                            cv2.circle(art_img, (cx, cy), radius, color_outer, cv2.FILLED)
                            cv2.circle(art_img, (cx, cy), max(1, int(radius * 0.4)), color_inner, cv2.FILLED)
                        # Intense blooming highlight
                        cv2.circle(art_img, (cx, cy), radius + 2, color_outer, 2)
            
            # --- Magic Button Interaction ---
            if style == "Magic Button":
                btn_sz = 100
                # Initialize button position if not set
                if self.button_pos is None:
                    # Default to Top-Right
                    self.button_pos = (w - btn_sz//2 - 30, 30 + btn_sz//2)
                
                bx, by = self.button_pos
                x1, y1 = bx - btn_sz//2, by - btn_sz//2
                x2, y2 = bx + btn_sz//2, by + btn_sz//2
                self.button_rect = (x1, y1, x2, y2)
                
                # Draw Button Region
                cv2.circle(art_img, (x1 + btn_sz//2, y1 + btn_sz//2), btn_sz//2 + 5, (0, 0, 200), -1) # Vivid Red Circle
                cv2.circle(art_img, (x1 + btn_sz//2, y1 + btn_sz//2), btn_sz//2 + 5, (255, 255, 255), 2) # White stroke
                
                # Overlay Button Image if loaded
                if self.btn_image_raw is not None:
                    btn_resized = cv2.resize(self.btn_image_raw, (btn_sz, btn_sz))
                    if btn_resized.shape[2] == 4:
                        for c in range(0, 3):
                            alpha_channel = btn_resized[:,:,3]/255.0
                            art_img[y1:y2, x1:x2, c] = btn_resized[:,:,c] * alpha_channel + \
                                                      art_img[y1:y2, x1:x2, c] * (1.0 - alpha_channel)
                    else:
                        art_img[y1:y2, x1:x2] = btn_resized

                # Check Collision (Wrist 15, 16)
                if self.magic_cooldown > 0:
                    self.magic_cooldown -= 1
                else:
                    for id in [15, 16]:
                        if id in points:
                            px, py = points[id]
                            if x1 < px < x2 and y1 < py < y2:
                                self.magic_button_active = not self.magic_button_active
                                self.magic_touch_count += 1
                                status = f"Touch #{self.magic_touch_count}!"
                                color = "pink" if self.magic_button_active else "gray"
                                self.status_label.configure(text=status, text_color=color)
                                self.magic_cooldown = 30 # 1.5 second cooldown
                                
                                # Move button to a random place
                                margin = btn_sz + 50
                                self.button_pos = (
                                    random.randint(margin, w - margin),
                                    random.randint(margin, h - margin)
                                )
                                
                                # Randomize Colors
                                self.magic_bg_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                                o_col = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                                i_col = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
                                self.magic_stickman_theme = {"outline": o_col, "inner": i_col, "joint": o_col}
                                
                                self.after(2000, lambda: self.status_label.configure(text=status if self.magic_cooldown > 0 else "Ready", text_color="gray"))
                                break
                
                # Draw Counter at Top Center
                counter_text = f"TOUCHES: {self.magic_touch_count}"
                font = cv2.FONT_HERSHEY_DUPLEX
                f_scale = 1.2
                f_thick = 2
                (tw, th), baseline = cv2.getTextSize(counter_text, font, f_scale, f_thick)
                tx = (w - tw) // 2
                ty = 60
                
                # Shadow/Glow effect
                cv2.putText(art_img, counter_text, (tx+2, ty+2), font, f_scale, (20, 20, 20), f_thick + 2)
                # Main text
                c_color = (255, 180, 255) if self.magic_button_active else (200, 200, 200)
                cv2.putText(art_img, counter_text, (tx, ty), font, f_scale, c_color, f_thick)
                # Underline
                cv2.line(art_img, (tx, ty + 10), (tx + tw, ty + 10), c_color, 2)
                    
            # Draw Joints (not for Minimalist Line Art)
            if style != "Minimalist Line Art" and node_sz > 0:
                joint_ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                for id in joint_ids:
                    if id in points:
                        cv2.circle(art_img, points[id], int(node_sz), theme["joint"], cv2.FILLED)
                        cv2.circle(art_img, points[id], int(node_sz), (255, 255, 255), max(1, int(inner_thickness/2)))

        # --- Minimalist Line Art: Real Body Sketch ---
        if style == "Minimalist Line Art":
            # Step 1: Background subtraction to isolate person
            fg_mask = self.bg_subtractor.apply(img, learningRate=0.005)
            # Clean up — larger kernels fill gaps around face/hair/hands
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
            fg_mask = cv2.GaussianBlur(fg_mask, (21, 21), 0)
            _, fg_mask = cv2.threshold(fg_mask, 60, 255, cv2.THRESH_BINARY)

            # Step 2: High-quality pencil sketch pipeline
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Triple-pass bilateral filter: smooths skin texture, preserves real edges
            smooth = gray
            for _ in range(3):
                smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=80, sigmaSpace=80)

            # Canny edge detection: crisp, clean, continuous outlines
            canny = cv2.Canny(smooth, threshold1=20, threshold2=55)
            # Slightly thicken lines for clear visibility
            canny = cv2.dilate(canny, np.ones((2, 2), np.uint8), iterations=1)

            # Invert: Canny gives white edges on black, we want black lines on white
            canny_inv = cv2.bitwise_not(canny)
            edges_bgr = cv2.cvtColor(canny_inv, cv2.COLOR_GRAY2BGR)

            # Step 3: Mask — white background, sketch only where person is
            edges_masked = np.where(
                fg_mask[:, :, None] > 0,
                edges_bgr,
                np.ones((h, w, 3), dtype=np.uint8) * 255
            ).astype(np.uint8)
            art_img = edges_masked

            # Step 4: Subtle trail / ghosting effect
            if self.miniline_canvas is None or self.miniline_canvas.shape[:2] != (h, w):
                self.miniline_canvas = np.ones((h, w, c), dtype=np.uint8) * 255
            self.miniline_canvas = cv2.addWeighted(art_img, 0.9, self.miniline_canvas, 0.1, 0)
            art_img = self.miniline_canvas.copy()

                                        
        return art_img

    def change_style(self, direction):
        """Cycle through styles: direction 1 for next, -1 for previous"""
        styles = self.style_menu.cget("values")
        current_style = self.style_menu.get()
        try:
            idx = styles.index(current_style)
        except ValueError:
            idx = 0
            
        new_idx = (idx + direction) % len(styles)
        new_style = styles[new_idx]
        self.style_menu.set(new_style)
        
        # Reset Magic Button state and Minimalist canvas when switching
        self.magic_button_active = False
        self.particles = []
        self.miniline_canvas = None
        
        self.status_label.configure(text=f"Status: Style -> {new_style}", text_color="cyan")
        # Pulse the status label color back to gray after a delay
        self.after(2000, lambda: self.status_label.configure(text_color="gray"))


    def detect_gesture(self, lm_list):
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return

        if not lm_list:
            return

        # Landmarks: 15 (left wrist), 16 (right wrist), 0 (nose), 11 (left shoulder), 12 (right shoulder)
        points = {lm[0]: (lm[1], lm[2]) for lm in lm_list}
        
        # Check for Both Hands Up (Wrists higher than Nose/Shoulders)
        if 15 in points and 16 in points:
            lw_y, rw_y = points[15][1], points[16][1]
            ref_y = points[0][1] if 0 in points else (points[11][1] if 11 in points else 200)
            
            # TRIGGER: Both Hands Up -> Change Style Once
            # Peak position detection: wrists significantly above nose/shoulders
            if lw_y < ref_y - 30 and rw_y < ref_y - 30: 
                current_time = time.time()
                if current_time - self.last_style_change_time > 3.0: # 3s Cooldown
                    self.change_style(1)
                    self.last_style_change_time = current_time
                    self.gesture_cooldown = 40 # Prevent rapid multi-triggers
                    self.status_label.configure(text=f"Status: Style Changed -> {self.style_menu.get()}", text_color="cyan")
                    self.after(2000, lambda: self.status_label.configure(text_color="gray"))

    def update_video(self):
        if not self.camera_running:
            return

        success, img = self.cap.read()
        if success:
            img = cv2.flip(img, 1) # Mirror image
            h, w, _ = img.shape
            self.frame_count += 1
            
            # --- Detection Logic ---
            style = self.style_menu.get()
            
            if style == "Ultimate 3D Wireframe":
                timestamp_ms = int(time.time() * 1000)
                # Skip face/hand inference every 2nd frame for perf
                skip = (self.frame_count % 2 == 0)
                face_results, hand_results = self.holistic_detector.process(img, timestamp_ms, skip_inference=skip)
                
                img_processed = img.copy()
                self.detector.find_pose(img_processed, timestamp_ms, draw=False)
                lm_list = self.detector.find_position(img_processed)
                
                self.holistic_face_results = face_results
                self.holistic_hand_results = hand_results
            else:
                draw_original = self.switch_draw_original.get() == 1
                img_processed = img.copy()
                timestamp_ms = int(time.time() * 1000)
                self.detector.find_pose(img_processed, timestamp_ms, draw=draw_original)
                lm_list = self.detector.find_position(img_processed)
                self.holistic_face_results = None
                self.holistic_hand_results = None
            
            # Gesture Detection
            if self.switch_gesture_control.get() == 1:
                self.detect_gesture(lm_list)
            else:
                self.gesture_history = [] # Clear history when disabled
            
            # Use fixed dimensions to maintain UI stability
            ui_width = 480
            
            # ---- Camera Feed Updates ----
            img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            ratio = ui_width / pil_img.width
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            
            ctk_img = ctk.CTkImage(light_image=pil_img, size=new_size)
            self.lbl_video.configure(image=ctk_img)
            self.lbl_video.image = ctk_img
            
            # ---- Stickman Art Updates ----
            if self.switch_draw_stickman.get() == 1:
                art_img = self.draw_stickman(img, lm_list)
                art_img_rgb = cv2.cvtColor(art_img, cv2.COLOR_BGR2RGB)
                pil_art = Image.fromarray(art_img_rgb)
                ctk_art = ctk.CTkImage(light_image=pil_art, size=new_size)
                self.lbl_art.configure(image=ctk_art)
                self.lbl_art.image = ctk_art
            else:
                self.lbl_art.configure(image=self._blank_img)

        self.after(20, self.update_video)

    def on_closing(self):
        self.camera_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
