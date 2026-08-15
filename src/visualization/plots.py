import matplotlib.pyplot as plt


def plot_raw_signals(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["time"],
        df["FBG1"],
        label="FBG1"
    )

    plt.plot(
        df["time"],
        df["FBG2"],
        label="FBG2"
    )

    plt.plot(
        df["time"],
        df["FBG3"],
        label="FBG3"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Bragg Wavelength (nm)")
    plt.title("Raw FBG Signals")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_wavelength_shift(
    time,
    wavelength_shift,
    channel
):

    plt.figure(figsize=(12, 5))

    plt.plot(
        time,
        wavelength_shift
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Wavelength Shift Δλ (nm)")

    plt.title(
        f"{channel} - Wavelength Shift"
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()
def plot_filter_comparison(
    time,
    raw_signal,
    filtered_signals,
    channel
):

    plt.figure(figsize=(14, 7))

    # Raw signal
    plt.plot(
        time,
        raw_signal,
        label="Raw",
        alpha=0.5
    )

    # Each filtered signal
    for name, signal in filtered_signals.items():

        plt.plot(
            time,
            signal,
            label=name
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Wavelength Shift Δλ (nm)")

    plt.title(
        f"{channel} - Filter Comparison"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()