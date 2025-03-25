import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np


class ImageEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Editor")

        self.original_image = None
        self.processed_image = None

        # Create the main GUI layout
        self.create_widgets()

        self.original_image = np.array(Image.open("test/imgs/dog.jpg"))
        self.processed_image = self.original_image.copy()
        self.display_image(self.original_image, self.original_label)
        self.display_image(self.processed_image, self.processed_label)

    def create_widgets(self):
        # Menu Bar #
        menubar = tk.Menu(self.master)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Save Image", command=self.save_image)
        menubar.add_cascade(label="File", menu=file_menu)

        # Operations menu
        operations_menu = tk.Menu(menubar, tearoff=0)
        operations_menu.add_command(label="Grayscale", command=self.to_grayscale)
        operations_menu.add_command(label="Negation", command=self.to_negative)
        operations_menu.add_command(label="Binarize", command=self.binarize_command)
        menubar.add_cascade(label="Operation", menu=operations_menu)

        # Filters menu
        filters_menu = tk.Menu(menubar, tearoff=0)
        filters_menu.add_command(label="Average Blur", command=self.filter_average)
        filters_menu.add_command(label="Gaussian Blur", command=self.filter_gaussian)
        filters_menu.add_command(label="Sharpen", command=self.filter_sharpen)
        filters_menu.add_command(label="Roberts", command=self.filter_roberts)
        filters_menu.add_command(label="Sobel", command=self.filter_sobel)
        filters_menu.add_command(label="Prewitt", command=self.filter_prewitt)
        filters_menu.add_command(label="Custom...", command=self.filter_custom)
        menubar.add_cascade(label="Filters", menu=filters_menu)

        # Plots menu
        plots_menu = tk.Menu(menubar, tearoff=0)
        plots_menu.add_command(label="Histograms", command=self.show_histogram)
        plots_menu.add_command(label="Projection vertical", command=self.show_projections_vertical)
        plots_menu.add_command(label="Projection horizontal", command=self.show_projections_horizontal)
        menubar.add_cascade(label="Plots", menu=plots_menu)

        # Set menubar
        self.master.config(menu=menubar)

        # #

        # - Content - #
        # Main Frame
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT for sliders #
        self.left_frame = tk.Frame(self.main_frame, width=200, height=500, bg="white")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.left_frame.pack_propagate(False)  # Keep fixed size

        tk.Label(self.left_frame, text="Adjustments", bg="white", fg="black", font=("Arial", 12, "bold")).pack(pady=10)

        # Brightness slider
        tk.Label(self.left_frame, text="Brightness", bg="white", fg="black").pack()
        self.brightness_slider = tk.Scale(self.left_frame, from_=-100, to=100, orient="vertical")
        self.brightness_slider.pack()

        # Contrast slider
        tk.Label(self.left_frame, text="Contrast", bg="white", fg="black").pack()
        self.contrast_slider = tk.Scale(self.left_frame, from_=-128, to=128, orient="vertical")
        self.contrast_slider.pack()

        # Apply brightness and contrast button
        self.apply_button = tk.Button(self.left_frame, text = "Apply", command=self.apply_contrast_and_brightness)
        self.apply_button.pack(pady=10)

        # MIDDLE frame for original image #
        self.middle_frame = tk.Frame(self.main_frame, width=400, height=400, bg="white", padx=10)
        self.middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.middle_frame.pack_propagate(False)  # Keep fixed size
        # Title for Original Image
        self.original_title = tk.Label(self.middle_frame, text="Original", bg="white", fg="black",
                                       font=("Arial", 12, "bold"))
        self.original_title.pack()
        # Label for Original Image
        self.original_label = tk.Label(self.middle_frame, text="No Image", bg="white")
        self.original_label.pack(fill=tk.BOTH, expand=True)

        # Frame for button between images
        self.middle_button_frame = tk.Frame(self.main_frame, width=50, bg="white")
        self.middle_button_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.middle_button_frame.pack_propagate(False)

        # Button to move processed image to original (placed between images)
        self.middle_apply_button = tk.Button(self.middle_button_frame, text="←",
                                             command=self.processed_to_original, font=("Arial", 14, "bold"))
        self.middle_apply_button.pack(expand=True, pady=10)

        # RIGHT frame for processed image #
        self.right_frame = tk.Frame(self.main_frame, width=400, height=400, bg="white", padx=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.right_frame.pack_propagate(False)  # Keep fixed size
        # Title for Processed Image
        self.processed_title = tk.Label(self.right_frame, text="Processed", bg="white", fg="black",
                                        font=("Arial", 12, "bold"))
        self.processed_title.pack()
        # Label for Processed Image
        self.processed_label = tk.Label(self.right_frame, text="No Image", bg="white")
        self.processed_label.pack(fill=tk.BOTH, expand=True)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"), ("All Files", "*.*")]
        )
        if file_path:
            self.original_image = Image.open(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Failed to open image!")
                return
            # Convert to RGB for consistent display in Tkinter
            self.original_image = np.array(self.original_image)
            self.processed_image = self.original_image.copy()
            self.display_image(self.original_image, self.original_label)
            self.display_image(self.processed_image, self.processed_label)

    def display_image(self, img, label):
        def resize_image(image, size=(400, 400)):
            img_pil = Image.fromarray(image)
            img_pil.thumbnail(size, Image.LANCZOS)
            return ImageTk.PhotoImage(img_pil)

        imgtk = resize_image(img)
        label.configure(image=imgtk)
        label.image = imgtk

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No image to save.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")]
        )
        if file_path:
            img = self.processed_image
            if len(self.processed_image.shape) == 2:  # Bcs this is needed for grayscale saving
                img = np.uint8(self.processed_image)

            save_img = Image.fromarray(img)
            save_img.save(file_path)
            messagebox.showinfo("Saved", f"Image saved to {file_path}")

    def processed_to_original(self):
        if self.processed_image is not None:
            self.original_image = self.processed_image.copy()
        self.display_image(self.original_image, self.original_label)

    def to_grayscale(self):
        if self.processed_image is None:
            return
        if len(self.original_image.shape) != 3:  # Don't grayscale if already grayscaled
            return
        self.processed_image = np.dot(self.original_image, np.array([0.2989, 0.5870, 0.1140]))
        self.display_image(self.processed_image, self.processed_label)

    def to_negative(self):
        if self.processed_image is None:
            return
        self.processed_image = 255 - self.original_image
        self.display_image(self.processed_image, self.processed_label)

    def binarize_command(self):
        threshold = simpledialog.askinteger("Binarization", "Enter threshold (0-255):", minvalue=0,
                                            maxvalue=255, initialvalue=128)
        if threshold is None:
            return
        self.binarize(threshold)

    def binarize(self, threshold=128):
        if self.processed_image is None:
            return

        if len(self.original_image.shape) == 3:  # Don't grayscale if already grayscaled
            self.processed_image = self.original_image.mean(axis=2)  # To grayscale
        else:
            self.processed_image = self.original_image.copy()

        self.processed_image = np.where(self.processed_image < threshold, 0, 255).astype(np.uint8)
        self.display_image(self.processed_image, self.processed_label)

    def apply_contrast_and_brightness(self):
        if self.original_image is None:
            return

        brightness = self.brightness_slider.get()
        contrast = self.contrast_slider.get()

        # Factor for contrast
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        img = np.array(self.original_image, dtype=np.int32)

        img = (factor * (img-128) + 128) + brightness
        img = np.clip(img, 0, 255)  # Ensure pixel values remain valid
        self.processed_image = np.array(img, dtype=self.original_image.dtype)
        self.display_image(self.processed_image, self.processed_label)

    def convolution_gray(self, img, kernel):
        # Define shape for extended image
        new_shape = list(img.shape)
        pad_top = kernel.shape[0] // 2
        pad_bottom = kernel.shape[0] - 1 - pad_top
        pad_left = kernel.shape[1] // 2
        pad_right = kernel.shape[1] - 1 - pad_left
        new_shape[0] += pad_top + pad_bottom
        new_shape[1] += pad_left + pad_right

        # Create extended image
        extended_img = np.full(new_shape, 0, dtype=img.dtype)

        # Fill center of and extended image with actual image
        extended_img[pad_top:(new_shape[0] - pad_bottom), pad_left:(new_shape[1] - pad_right)] = img.copy()

        # Fill paddings of extended image
        # UP
        arr = img[0, :]
        extended_img[:pad_top, pad_left:(new_shape[1] - pad_right)] = np.repeat(arr[np.newaxis, :], pad_top, axis=0)
        # DOWN
        arr = img[-1, :]
        extended_img[(new_shape[0] - pad_bottom):, pad_left:(new_shape[1] - pad_right)] = np.repeat(arr[np.newaxis, :],
                                                                                                    pad_bottom, axis=0)
        # LEFT
        arr = img[:, 0]
        extended_img[pad_top:(new_shape[0] - pad_bottom), :pad_left] = np.repeat(arr[:, np.newaxis], pad_left, axis=1)
        # RIGHT
        arr = img[:, -1]
        extended_img[pad_top:(new_shape[0] - pad_bottom), (new_shape[1] - pad_right):] = np.repeat(arr[:, np.newaxis],
                                                                                                   pad_right, axis=1)
        # UP-LEFT CORNER
        extended_img[:pad_top, :pad_left] = np.full((pad_top, pad_left), img[0, 0])
        # UP-RIGHT CORNER
        extended_img[:pad_top, (new_shape[1] - pad_right):] = np.full((pad_top, pad_right), img[0, -1])
        # DOWN-LEFT CORNER
        extended_img[(new_shape[0] - pad_bottom):, :pad_left] = np.full((pad_bottom, pad_left), img[-1, 0])
        # DOWN-RIGHT CORNER
        extended_img[(new_shape[0] - pad_bottom):, (new_shape[1] - pad_right):] = np.full((pad_bottom, pad_right),
                                                                                          img[-1, -1])

        # Do the convolution
        result = np.full(img.shape, 0, dtype=img.dtype)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                fragment = extended_img[i:(i + kernel.shape[0]),
                           j:(j + kernel.shape[1])]  # fragment of img which is convoluted in this iteration
                pixel = np.sum(fragment * kernel)
                pixel = np.clip(pixel, a_min=0,
                                a_max=255)  # clip to range 0 - 255 if it exceeds (0 when number<0 and 255 when number>255
                result[i, j] = pixel

        print("Done")
        return result

    def convolution(self, kernel):
        if self.original_image.ndim == 2:  # Grayscale image
            return self.convolution_gray(self.original_image, kernel)
        elif self.original_image.ndim == 3:  # Color image
            H, W, C = self.original_image.shape
            result = np.zeros_like(self.original_image)
            # Convolve r, g, b separately
            for c in range(C):
                result[:, :, c] = self.convolution_gray(self.original_image[:, :, c], kernel)
            return result
        else:
            raise ValueError("Unsupported image dimensions")

    def filter_average(self):
        kernel = np.full((5, 5), 1/25)
        self.processed_image = self.convolution(kernel)
        self.display_image(self.processed_image, self.processed_label)

    def filter_gaussian(self):
        kernel = np.array([[1,  4,  7,  4, 1],
                                 [4, 16, 26, 16, 4],
                                 [7, 26, 41, 26, 7],
                                 [4, 16, 26, 16, 4],
                                 [1,  4,  7,  4, 1]], dtype=np.float32)
        kernel /= kernel.sum()
        self.processed_image = self.convolution(kernel)
        self.display_image(self.processed_image, self.processed_label)

    def filter_sharpen(self):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        self.processed_image = self.convolution(kernel)
        self.display_image(self.processed_image, self.processed_label)

    def filter_custom(self):

        def get_matrix_from_entries(entries, rows, cols):
            return np.array([[eval(entries[r][c].get()) for c in range(cols)] for r in range(rows)])

        def update_matrix_display(root, frame, entries, rows, cols):
            for widget in frame.winfo_children():
                widget.destroy()

            new_entries = []
            for r in range(rows):
                row_entries = []
                for c in range(cols):
                    var = tk.StringVar(value="1")
                    entry = ttk.Entry(frame, width=5, textvariable=var)
                    entry.grid(row=r, column=c, padx=2, pady=2)
                    if r < len(entries) and c < len(entries[r]):
                        var.set(entries[r][c].get())
                    row_entries.append(var)
                new_entries.append(row_entries)

            return new_entries

        def matrix_input_window():
            root = tk.Toplevel()
            root.title("Kernel Input")

            frame = ttk.Frame(root)
            frame.pack(pady=10)

            entries = []
            rows, cols = 3, 3  # Default matrix size
            matrix_var = tk.Variable()

            def increase_rows():
                nonlocal rows
                rows += 1
                nonlocal entries
                entries = update_matrix_display(root, frame, entries, rows, cols)

            def decrease_rows():
                nonlocal rows
                if rows > 1:
                    rows -= 1
                    nonlocal entries
                    entries = update_matrix_display(root, frame, entries, rows, cols)

            def increase_cols():
                nonlocal cols
                cols += 1
                nonlocal entries
                entries = update_matrix_display(root, frame, entries, rows, cols)

            def decrease_cols():
                nonlocal cols
                if cols > 1:
                    cols -= 1
                    nonlocal entries
                    entries = update_matrix_display(root, frame, entries, rows, cols)

            def get_matrix():
                matrix = get_matrix_from_entries(entries, rows, cols)
                matrix_var.set(matrix.tolist())
                root.destroy()

            entries = update_matrix_display(root, frame, entries, rows, cols)

            control_frame = ttk.Frame(root)
            control_frame.pack(pady=5)

            ttk.Button(control_frame, text="+ Row", command=increase_rows).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="- Row", command=decrease_rows).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="+ Col", command=increase_cols).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="- Col", command=decrease_cols).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="OK", command=get_matrix).pack(side=tk.LEFT, padx=5)

            root.wait_window()  # Pause execution until the window is closed
            return np.array(matrix_var.get())

        kernel = matrix_input_window()
        if kernel is not None:
            self.processed_image = self.convolution(kernel)
            self.display_image(self.processed_image, self.processed_label)

    def filter_roberts(self):
        kernel1 = np.array([[1, 0],
                            [0, -1]])
        kernel2 = np.array([[0, 1],
                            [-1, 0]])

        if len(self.original_image.shape) == 3:
            img = self.original_image.mean(axis=2)
            conv1 = self.convolution_gray(img, kernel1)
            conv2 = self.convolution_gray(img, kernel2)
        else:
            conv1 = self.convolution_gray(self.original_image, kernel1)
            conv2 = self.convolution_gray(self.original_image, kernel2)

        self.processed_image = (conv1**2 + conv2**2)**(1/2)
        self.display_image(self.processed_image, self.processed_label)

    def filter_sobel(self):
        kernel1 = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        kernel2 = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        if len(self.original_image.shape) == 3:
            img = self.original_image.mean(axis=2)
            conv1 = self.convolution_gray(img, kernel1)
            conv2 = self.convolution_gray(img, kernel2)
        else:
            conv1 = self.convolution_gray(self.original_image, kernel1)
            conv2 = self.convolution_gray(self.original_image, kernel2)
        self.processed_image = (conv1 ** 2 + conv2 ** 2) ** (1 / 2)
        self.display_image(self.processed_image, self.processed_label)

    def filter_prewitt(self):
        kernel1 = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        kernel2 = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]])
        if len(self.original_image.shape) == 3:
            img = self.original_image.mean(axis=2)
            conv1 = self.convolution_gray(img, kernel1)
            conv2 = self.convolution_gray(img, kernel2)
        else:
            conv1 = self.convolution_gray(self.original_image, kernel1)
            conv2 = self.convolution_gray(self.original_image, kernel2)
        self.processed_image = (conv1 ** 2 + conv2 ** 2) ** (1 / 2)
        self.display_image(self.processed_image, self.processed_label)

    def show_histogram(self):
        if self.original_image is None:
            return

        if len(self.processed_image.shape) == 3:  # For color images
            gray_img = self.processed_image.mean(axis=2)
            fig, axs = plt.subplots(2, 2)
            axs = axs.flatten()

            # Gray histogram
            axs[0].hist(gray_img.ravel(), bins=256, range=(0, 255), color='black')
            axs[0].set_title("Grayscale Histogram")
            axs[0].set_xlabel("Pixel Intensity")
            axs[0].set_ylabel("Frequency")
            # Red canal histogram
            axs[1].hist(self.processed_image[:, :, 0].ravel(), bins=256, range=(0, 255), color='red')
            axs[1].set_title("Red Canal Histogram")
            axs[1].set_xlabel("Red Intensity")
            axs[1].set_ylabel("Frequency")
            # Green canal histogram
            axs[2].hist(self.processed_image[:, :, 1].ravel(), bins=256, range=(0, 255), color='green')
            axs[2].set_title("Green Canal Histogram")
            axs[2].set_xlabel("Green Intensity")
            axs[2].set_ylabel("Frequency")
            # Blue canal histogram
            axs[3].hist(self.processed_image[:, :, 2].ravel(), bins=256, range=(0, 255), color='blue')
            axs[3].set_title("Blue Canal Histogram")
            axs[3].set_xlabel("Blue Intensity")
            axs[3].set_ylabel("Frequency")

            plt.tight_layout()
            plt.show()
        if len(self.processed_image.shape) == 2:  # For gray images
            # Gray histogram
            plt.hist(self.processed_image.ravel(), bins=256, range=(0, 255), color='black')
            plt.title("Grayscale Histogram")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")

            plt.show()

    def show_projections_vertical(self):
        if self.original_image is None:
            return

        if len(self.processed_image.shape) == 3:  # For color images
            gray_img = self.processed_image.mean(axis=2)
            fig, axs = plt.subplots(2, 2)
            axs = axs.flatten()
            # Gray projection
            axs[0].hlines(xmin=0, xmax=gray_img.sum(axis=1), y=range(gray_img.shape[0]), color="black", alpha=0.4)
            axs[0].invert_yaxis()
            axs[0].set_title("Grayscale")
            axs[0].set_xlabel("Sum of Row Intensity")
            axs[0].set_ylabel("Y coordinate")
            # Red canal projection
            axs[1].hlines(xmin=0, xmax=self.processed_image[:, :, 0].sum(axis=1), y=range(gray_img.shape[0]), color="red", alpha=0.4)
            axs[1].invert_yaxis()
            axs[1].set_title("Red Canal")
            axs[1].set_xlabel("Sum of Row Intensity")
            axs[1].set_ylabel("Y coordinate")
            # Green canal projection
            axs[2].hlines(xmin=0, xmax=self.processed_image[:, :, 1].sum(axis=1), y=range(gray_img.shape[0]), color="green", alpha=0.4)
            axs[2].invert_yaxis()
            axs[2].set_title("Green Canal")
            axs[2].set_xlabel("Sum of Row Intensity")
            axs[2].set_ylabel("Y coordinate")
            # Blue canal projection
            axs[3].hlines(xmin=0, xmax=self.processed_image[:, :, 2].sum(axis=1), y=range(gray_img.shape[0]), color="blue", alpha=0.4)
            axs[3].invert_yaxis()
            axs[3].set_title("Blue Canal")
            axs[3].set_xlabel("Sum of Row Intensity")
            axs[3].set_ylabel("Y coordinate")
        else:  # Grayscale
            plt.figure()
            ax = plt.axes()
            plt.hlines(xmin=0, xmax=self.processed_image.sum(axis=1), y=range(self.processed_image.shape[0]), color="black", alpha=0.4)
            ax.invert_yaxis()
            plt.title("Grayscale")
            plt.xlabel("Sum of Row Intensity")
            plt.ylabel("Y coordinate")
        plt.tight_layout()
        plt.show()

    def show_projections_horizontal(self):
        if self.original_image is None:
            return

        if len(self.processed_image.shape) == 3:  # For color images
            gray_img = self.processed_image.mean(axis=2)
            fig, axs = plt.subplots(2, 2)
            axs = axs.flatten()
            # Gray projection
            axs[0].vlines(ymin=0, ymax=gray_img.sum(axis=0), x=range(gray_img.shape[1]), color="black", alpha=0.4)
            axs[0].set_title("Grayscale")
            axs[0].set_xlabel("X coordinate")
            axs[0].set_ylabel("Sum of Column Intensity")
            # Red canal projection
            axs[1].vlines(ymin=0, ymax=self.processed_image[:, :, 0].sum(axis=0), x=range(gray_img.shape[1]), color="red", alpha=0.4)
            axs[1].set_title("Red Canal")
            axs[1].set_xlabel("X coordinate")
            axs[1].set_ylabel("Sum of Column Intensity")
            # Green canal projection
            axs[2].vlines(ymin=0, ymax=self.processed_image[:, :, 1].sum(axis=0), x=range(gray_img.shape[1]), color="green", alpha=0.4)
            axs[2].set_title("Green Canal")
            axs[2].set_xlabel("X coordinate")
            axs[2].set_ylabel("Sum of Column Intensity")
            # Blue canal projection
            axs[3].vlines(ymin=0, ymax=self.processed_image[:, :, 2].sum(axis=0), x=range(gray_img.shape[1]), color="blue", alpha=0.4)
            axs[3].set_title("Blue Canal")
            axs[3].set_xlabel("X coordinate")
            axs[3].set_ylabel("Sum of Column Intensity")
        else:  # Grayscale
            plt.figure()
            ax = plt.axes()
            ax.vlines(ymin=0, ymax=self.processed_image.sum(axis=0), x=range(self.processed_image.shape[1]), color="black", alpha=0.4)
            ax.set_title("Grayscale")
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Sum of Column Intensity")

        plt.tight_layout()
        plt.show()




