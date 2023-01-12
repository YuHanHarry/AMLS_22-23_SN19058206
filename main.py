# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def menu(selection):
    if selection == 'A1':
        import A1.gender
        A1.gender()
    if selection == 'A2':
        import A2.smiling
        A2.smiling()
    if selection == 'B1':
        import B1.FaceShape
        B1.FaceShape()
    if selection == 'B2':
        import B2.EyeColor
        B2.EyeColor()

if __name__ == '__main__':
    selection = input("Please select the task: A1 A2 B1 B2:")
    menu(selection)


