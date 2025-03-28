import math
import tabulate
import matplotlib.pyplot as plt

files = ["Act 4.1/data01.txt", "Act 4.1/data02.txt", "Act 4.1/data03.txt"]
decimals = [4, 2, 3]

for i, file in enumerate(files):
    with open(file, "r") as f:
        lines = f.readlines()
        n = len(lines)
        c = math.ceil(1 + 3.3 * math.log10(n))

        data = [float(line.strip()) for line in lines]

        max_value = round(max(data), decimals[i])
        min_value = round(min(data), decimals[i])

        w = round((max_value - min_value) / c, decimals[i])

        intervals = [(min_value + i * w, min_value + (i + 1) * w) for i in range(c)]
        freq_table = {interval: 0 for interval in intervals}

        for value in data:
            for interval in intervals:
                if interval[0] <= value < interval[1]:
                    freq_table[interval] += 1

        print()
        print(f"Data File: {i + 1}")
        print(f"N: {n}")
        print(f"C: {c}")
        print(f"Max: {max_value}, Min: {min_value}")
        print(f"W: {w}")
        print()
        print(
            tabulate.tabulate(
                freq_table.items(), headers=["Interval", "Frequency"], tablefmt="grid"
            )
        )
        print()
        print("Sum of Frequencies:", sum(freq_table.values()))
        print()

        plt.bar(
            [f"{interval[0]} - {interval[1]}" for interval in intervals],
            freq_table.values(),
        )
        plt.xlabel("Interval")
        plt.ylabel("Frequency")
        plt.title(f"Data File {i + 1}")
        plt.show()
