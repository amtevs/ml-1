import os
import argparse
import numpy as np

from time import time
from hw5code import find_best_split, DecisionTree


def main(strict, throw_error):
    start = time()
    for task_num, q in enumerate(sorted(os.listdir("A"))):
        d = np.load(f"A/{q}", allow_pickle=True)
        globals().update(d)
        right_solutions = thresholds, ginis, threshold_best, gini_best
        your_answers = find_best_split(feature_vector, target_vector)
        labels = ["thesholds", "ginis", "threshold_best", "gini_best"]
        for your_answer, right_solution, label in zip(your_answers, right_solutions, labels):

            error_message = f"Неправильный {label} в задаче {task_num + 1}.\n{feature_vector=}\n{target_vector=}\n{your_answer=}\n{right_solution=}"
            if hasattr(your_answer, "shape"):
                assert (
                    np.float64(your_answer).shape == np.float64(right_solution).shape
                ), f"{label}\n{your_answer.shape=}, {right_solution.shape=}"
            assert np.allclose(your_answer, right_solution), error_message
    end = time()
    print("Всё корректно в задаче A (функция find_best_split).")
    print(f"Тесты завершились за {end - start:.4f} секунд. (У автора тестов 0.0802)")

    start = time()
    errors = 0
    for task_num, q in enumerate(sorted(os.listdir("B"))):
        if task_num >= 15 and not strict:
            break
        globals().update(np.load(f"B/{q}", allow_pickle=True))
        your_tree = DecisionTree(feature_types)
        your_tree.fit(x_train, y_train.flatten())
        your_answer = your_tree.predict(x_test)
        error_message = f"Неправильный ответ в задаче {task_num+1}\n{your_answer=}\n{expected=}"
        assert your_answer.shape == expected.shape, f"{your_answer.shape=}, {expected.shape=}"
        try:
            assert (
                np.allclose(your_answer, expected, rtol=1e-4)
                or np.allclose(your_answer, expected2)
                or np.allclose(your_answer, expected3)
                or np.allclose(your_answer, expected4)
            ), error_message
        except AssertionError as e:
            if throw_error:
                raise e
            else:
                print(e)
                errors += 1
    end = time()
    if throw_error:
        print(f"Всё корректно в задаче B (класс DecisionTree)")
    else:
        print(f"Кол-во ошибок: {errors} в задаче B (класс DecisionTree)")
    print(f"Тесты завершились за {end - start:.2f} секунд. (У автора тестов 12.28 для всех задач)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Чекнуть реализацию решающего дерева локально.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Точная проверка всех решений на соответствие авторским если True, либо проверка первых 15 если False.",
    )
    parser.add_argument(
        "--throw_error",
        action="store_true",
        help="Завершать ли проверку при обнаружении ошибки (по умолчанию - да).",
    )
    args = parser.parse_args()
    main(strict=args.strict, throw_error=args.throw_error)
