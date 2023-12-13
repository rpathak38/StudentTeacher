import papermill as pm


def run_notebooks():
    # Execute the first notebook
    pm.execute_notebook(
        'student_train_alone.ipynb',
        'output_student_train_alone.ipynb'  # Output notebook for the first notebook
    )

    # Execute the second notebook
    pm.execute_notebook(
        'student_train_with_teacher.ipynb',
        'output_student_train_with_teacher.ipynb'  # Output notebook for the second notebook
    )


if __name__ == "__main__":
    run_notebooks()
