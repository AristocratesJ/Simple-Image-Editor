import tkinter as tk
from src.app.app import ImageEditor
def main():
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()