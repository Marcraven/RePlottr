import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import json
import time
from params import (
    TRAIN_SIZE,
    VAL_SPLIT,
    TEST_SPLIT,
    XLIM_LOW,
    XLIM_HIGH,
    YLIM_LOW,
    YLIM_HIGH,
    NUM_SERIES_MIN,
    NUM_SERIES_MAX,
    NUM_POINTS_MIN,
    NUM_POINTS_MAX,
    START_INDEX,
    FIGSIZE_WIDTH,
    FIGSIZE_HEIGHT,
    FIGSIZE_DPI,
)

##### Define constants ####
labels = [
    "Length [nm]",
    "Size [um]",
    "Length [mm]",
    "Size [cm]",
    "Distance [m]",
    "Distance [km]",
    "Weight [kg]",
    "Temperature [K]",
    "Temperature [C]",
    "Area [m^2]",
    "Area [cm^2]",
    "Volume [m^3]",
    "Volume [cm^3]",
    "Speed [m/s]",
    "Speed [km/h]",
    "Time [s]",
    "Time [min]",
    "Time [hr]",
    "Frequency [Hz]",
    "Energy [J]",
    "Power [W]",
    "Pressure [Pa]",
    "Pressure [kPa]",
    "Voltage [V]",
    "Current [A]",
    "Resistance [ohm]",
    "Capacitance [F]",
    "Inductance [H]",
    "Force [N]",
    "Torque [Nm]",
    "Velocity [m/s]",
    "Acceleration [m/s^2]",
    "Density [kg/m^3]",
    "Density [g/cm^3]",
    "Viscosity [Pa.s]",
    "Viscosity [cP]",
    "Flow rate [m^3/s]",
    "Flow rate [L/min]",
    "Concentration [mol/m^3]",
    "Concentration [mg/L]",
    "Luminance [cd/m^2]",
    "Illuminance [lux]",
    "Magnetic field [T]",
    "Magnetic flux [Wb]",
    "Radiation dose [Gy]",
    "Radiation dose rate [Gy/s]",
    "Angle [rad]",
    "Angle [deg]",
]

adjectives = [
    "Happy",
    "Sad",
    "Excited",
    "Mysterious",
    "Playful",
    "Enthusiastic",
    "Cautious",
    "Brave",
    "Shy",
    "Energetic",
    "Lazy",
    "Curious",
    "Confident",
    "Clumsy",
    "Grateful",
    "Generous",
    "Ambitious",
    "Compassionate",
    "Wise",
    "Silly",
    "Serious",
    "Sincere",
    "Gentle",
    "Tough",
    "Carefree",
    "Charming",
    "Radiant",
    "Elegant",
    "Witty",
    "Artistic",
    "Dynamic",
    "Resilient",
    "Daring",
    "Friendly",
    "Loyal",
    "Mellow",
    "Vibrant",
    "Vivacious",
    "Creative",
    "Humble",
    "Sassy",
    "Reckless",
    "Pensive",
    "Candid",
    "Adaptable",
    "Tenacious",
    "Resourceful",
    "Modest",
    "Charismatic",
    "Nurturing",
    "Fierce",
    "Optimistic",
    "Pessimistic",
    "Eccentric",
    "Charming",
    "Dazzling",
    "Bewildered",
    "Spirited",
    "Relaxed",
    "Cooperative",
    "Outgoing",
    "Introverted",
    "Quirky",
    "Sensitive",
    "Inquisitive",
    "Stoic",
    "Dramatic",
    "Whimsical",
    "Stoic",
    "Zesty",
    "Thoughtful",
    "Exuberant",
    "Spontaneous",
    "Candid",
    "Modest",
    "Confused",
    "Tenacious",
    "Resourceful",
    "Determined",
    "Jovial",
    "Playful",
    "Adventurous",
    "Reserved",
    "Hilarious",
    "Perceptive",
    "Easygoing",
    "Observant",
    "Reckless",
    "Enigmatic",
    "Witty",
    "Daring",
    "Spirited",
    "Hopeful",
    "Caring",
    "Bold",
    "Sincere",
    "Chivalrous",
    "Dynamic",
    "Courageous",
    "Grounded",
]

nouns = [
    "Sun",
    "Moon",
    "Ocean",
    "Mountain",
    "River",
    "Tree",
    "Cloud",
    "Bird",
    "Fish",
    "Flower",
    "Star",
    "Book",
    "Pen",
    "Key",
    "Door",
    "Window",
    "Table",
    "Chair",
    "Lamp",
    "Phone",
    "Computer",
    "Car",
    "Bicycle",
    "Train",
    "Plane",
    "City",
    "Country",
    "Friend",
    "Family",
    "Love",
    "Dream",
    "Adventure",
    "Journey",
    "Song",
    "Dance",
    "Art",
    "Science",
    "Math",
    "History",
    "Future",
    "Past",
    "Present",
    "Hope",
    "Fear",
    "Courage",
    "Wisdom",
    "Knowledge",
    "Truth",
    "Lie",
    "Freedom",
    "Justice",
    "Peace",
    "War",
    "Happiness",
    "Sadness",
    "Joy",
    "Sorrow",
    "Laughter",
    "Tear",
    "Smile",
    "Frown",
    "Success",
    "Failure",
    "Victory",
    "Defeat",
    "Challenge",
    "Reward",
    "Risk",
    "Adventure",
    "Discovery",
    "Wonder",
    "Imagination",
    "Creativity",
    "Innovation",
    "Silence",
    "Noise",
    "Nature",
    "Cityscape",
    "Island",
    "Desert",
    "Forest",
    "Meadow",
    "Valley",
    "Castle",
    "Kingdom",
    "Village",
    "Market",
    "Cafe",
    "Restaurant",
    "Library",
    "School",
    "Hospital",
    "Factory",
    "Office",
    "Home",
    "Ship",
    "Rocket",
    "Planet",
    "Galaxy",
    "Universe",
]

legends = [
    "Quantum",
    "Aether",
    "Nucleus",
    "Neutron",
    "Photon",
    "Spectrum",
    "Catalyst",
    "Polarity",
    "Ionization",
    "Isotope",
    "Helix",
    "Luminous",
    "Kinetics",
    "Entropy",
    "Synthesis",
    "Amino",
    "Genome",
    "Proton",
    "Orbit",
    "Radiance",
    "Cognate",
    "Inertia",
    "Momentum",
    "Inversion",
    "Fluorescence",
    "Catalysis",
    "Receptor",
    "Polymer",
    "Chromatin",
    "Matrix",
    "Ergonomics",
    "Thermodynamics",
    "Zeta",
    "Delta",
    "Alpha",
    "Beta",
    "Sigma",
    "Gamma",
    "Omega",
    "Psi",
    "Kappa",
    "Lambda",
    "Omicron",
    "Epsilon",
    "Enzyme",
    "Quasar",
    "Hydrogen",
    "Oxygen",
    "Astron",
    "Electron",
    "Plasma",
    "Cortex",
    "Neuron",
    "Entropy",
    "Eclipse",
    "Asteroid",
    "Kinematics",
]

markers = [
    ".",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "D",
    "d",
]

dot_colors = [
    "b",  # Single-letter abbreviations for basic colors
    "g",
    "r",
    "c",
    "m",
    "k",
    "blue",  # Full color names
    "green",
    "red",
    "cyan",
    "magenta",
    "black",
    "skyblue",
    "tomato",
    "gold",
    "purple",
    "lime",
    "orange",
    "pink",
    "brown",
    "gray",
    "#FF5733",  # Hexadecimal color values
    "#33FF57",
    "#5733FF",
    (0.1, 0.2, 0.3),  # RGB tuples
    (0.4, 0.5, 0.6),
    (0.7, 0.8, 0.9),
]

background_colors = ["white", "#F5F5F5", "#D3D3D3"]


##### Define data creation function #####
def create_data(start, end, folder):
    """This generates a number of Train, Evaluate or Test data on a specific folder
    start: number of first scatterplot picture
    end: number of last scatterplot picture
    folder: location for saving final pictures
    """
    metadata_list = []
    for j in range(start, end):
        # Generate random data
        xlim = np.random.randint(low=XLIM_LOW, high=XLIM_HIGH, size=1)
        ylim = np.random.randint(low=YLIM_LOW, high=YLIM_HIGH, size=1)
        num_series = (
            NUM_SERIES_MIN
            if NUM_SERIES_MIN == NUM_SERIES_MAX
            else np.random.randint(NUM_SERIES_MIN, NUM_SERIES_MAX)
        )
        num_points = np.random.randint(NUM_POINTS_MIN, NUM_POINTS_MAX)

        # Create an empty list to store series data
        series = []

        # Define figure and ax
        background_color = random.choice(background_colors)
        fig, ax = plt.subplots(
            figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT),
            facecolor=background_color,
            dpi=FIGSIZE_DPI,
        )

        # Generate and plot random data for each series
        for i in range(num_series):
            # Create series
            name = random.choice(legends)
            points_serie = max(num_points // num_series + np.random.randint(-2, 2), 1)
            x_data = np.round(np.random.rand(points_serie) * xlim, decimals=1)
            y_data = np.round(np.random.rand(points_serie) * ylim, decimals=1)
            marker = random.choice(markers)
            series.append(
                {"name": name, "marker": marker, "x": list(x_data), "y": list(y_data)}
            )

            # Create a scatter plot for the current series
            ax.scatter(
                x=x_data,
                y=y_data,
                label=name,
                marker=marker,
                color=random.choice(dot_colors),
            )

        # Add legend
        # ax.legend(loc="upper right", framealpha=0.3)  # , bbox_to_anchor=(0.6,0.5))

        # Add labels and title
        x_label = random.choice(labels)
        y_label = random.choice(labels)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plot_title = random.choice(adjectives) + " " + random.choice(nouns)
        ax.set_title(plot_title)
        ax.set_facecolor(background_color)

        fig.tight_layout()

        # Create file names

        fname = folder + str(j).zfill(4)

        # Save the plot with smaller margins
        fig.savefig(
            fname + ".jpg",
            dpi=FIGSIZE_DPI,
        )

        # Obtain ticks data
        x_ticks_data = ax.get_xticks()
        y_ticks_data = ax.get_yticks()

        # Get minimum axis values
        x_lim_min, x_lim_max = ax.get_xlim()
        y_lim_min, y_lim_max = ax.get_ylim()

        # Remove first and last tick items that are not visualised on chart
        if x_ticks_data.min() != x_lim_min:
            x_ticks_data = np.delete(
                x_ticks_data, np.where(x_ticks_data == x_ticks_data.min())
            )

        if x_ticks_data.max() != x_lim_max:
            x_ticks_data = np.delete(
                x_ticks_data, np.where(x_ticks_data == x_ticks_data.max())
            )

        if y_ticks_data.min() != y_lim_min:
            y_ticks_data = np.delete(
                y_ticks_data, np.where(y_ticks_data == y_ticks_data.min())
            )

        if y_ticks_data.max() != y_lim_max:
            y_ticks_data = np.delete(
                y_ticks_data, np.where(y_ticks_data == y_ticks_data.max())
            )

        # Create ground truth dictionary for DONUT and add it to metadata
        ground_truth = {
            "title": plot_title,
            "x_label": x_label,
            "x_ticks": list(x_ticks_data),
            "y_label": y_label,
            "y_ticks": list(y_ticks_data),
            # "series": series,
        }

        # Create metadata
        metadata = {
            "file_name": str(j).zfill(4) + ".jpg",
            "ground_truth": '{"gt_parse": ' + json.dumps(ground_truth) + "}",
        }
        metadata_list.append(metadata)

        ##### Define Yolo target #####
        yolo_target = np.empty((0, 5))

        ##### Add position coordinates of x axis ticks and y axis ticks to Yolo putput #####

        # Convert data coordinates to pixel coordinates
        x_ticks_pixel = ax.transData.transform([[x, y_lim_min] for x in x_ticks_data])
        y_ticks_pixel = ax.transData.transform([[x_lim_min, y] for y in y_ticks_data])

        # Split pixel coordinats into separate lists
        x_ticks_x_coord_pixel = x_ticks_pixel[:, 0]
        x_ticks_y_coord_pixel = x_ticks_pixel[:, 1]

        y_ticks_x_coord_pixel = y_ticks_pixel[:, 0]
        y_ticks_y_coord_pixel = y_ticks_pixel[:, 1]

        # Flip y coordinates verically
        fig_width, fig_height = fig.canvas.get_width_height()

        x_ticks_y_coord_pixel_flip = fig_height - x_ticks_y_coord_pixel
        y_ticks_y_coord_pixel_flip = fig_height - y_ticks_y_coord_pixel

        # Standardise pixel coordinates to (0,1)
        x_ticks_x_coord_std = x_ticks_x_coord_pixel / fig_width
        x_ticks_y_coord_std = (
            x_ticks_y_coord_pixel_flip / fig_height + 0.05  # Note manual adjustment
        )

        y_ticks_x_coord_std = (
            y_ticks_x_coord_pixel / fig_width - 0.05  # Note manual adjustment
        )
        y_ticks_y_coord_std = y_ticks_y_coord_pixel_flip / fig_height

        # Create additional rows to Yolo output
        yolo_target = np.vstack(
            (
                yolo_target,
                np.hstack(
                    (
                        0 * np.ones((len(x_ticks_data), 1)),
                        np.expand_dims(x_ticks_x_coord_std, 1),
                        np.expand_dims(x_ticks_y_coord_std, 1),
                        0.1 * np.ones((len(x_ticks_data), 2)),
                    )
                ),
                np.hstack(
                    (
                        1 * np.ones((len(y_ticks_data), 1)),
                        np.expand_dims(y_ticks_x_coord_std, 1),
                        np.expand_dims(y_ticks_y_coord_std, 1),
                        0.1 * np.ones((len(y_ticks_data), 2)),
                    )
                ),
            )
        )

        ##### Add dots' position coordinates to Yolo output #####
        for i in range(num_series):
            ### Obtain series data
            x_data = series[i]["x"]
            y_data = series[i]["y"]
            marker = series[i]["marker"]

            # Transform data to display pixel coordinates
            xy_display = ax.transData.transform(np.column_stack((x_data, y_data)))

            # Transform pixel coordinates to figure coordinates and flip y coordinates vertically
            xy_figure = fig.transFigure.inverted().transform(xy_display)

            # Flip the y axis because Yolo wants the 0,0 to be on the top left of the image
            xy_figure[:, 1] = 1 - xy_figure[:, 1]

            # Create additional rows to Yolo output
            len_xy = xy_figure.shape[0]
            yolo_target = np.vstack(
                (
                    yolo_target,
                    np.hstack(
                        (
                            np.zeros((len_xy, 1)) + markers.index(marker) + 2,
                            xy_figure,
                            0.05 * np.ones((len_xy, 2)),
                        )
                    ),
                )
            )

        # Save Yolo target in txt format that is read by Yolo model
        np.savetxt(fname + ".txt", yolo_target, delimiter=" ", fmt="%1.4f")

        # Clean figure, axes and close figure
        plt.clf()
        plt.cla()
        plt.close()

    # File path for the JSONL file
    file_path = folder + "/metadata.jsonl"

    # Writing data to the JSONL file
    with open(file_path, "w") as file:
        for item in metadata_list:
            json.dump(item, file, default=str)  # Use str() for non-serializable objects
            file.write("\n")  # Add a newline character to separate JSON objects


##### If name = main #####
if __name__ == "__main__":
    # Record start time
    start_time = time.time()

    # Turn interactive mode off
    plt.ioff()

    # Define arguments and folders
    if len(sys.argv) > 1:
        TRAIN_SIZE = int(sys.argv[1])

    data_dir = "./data"
    train_dir = data_dir + "/train/"
    val_dir = data_dir + "/validation/"
    test_dir = data_dir + "/test/"

    # Creat folders and generate files
    print("Starting training data creation...")
    os.makedirs(train_dir, exist_ok=True) if not os.path.exists(train_dir) else None
    create_data(START_INDEX, START_INDEX + TRAIN_SIZE, train_dir)

    print("Starting validation data creation...")
    os.makedirs(val_dir, exist_ok=True) if not os.path.exists(val_dir) else None
    create_data(START_INDEX, START_INDEX + int(TRAIN_SIZE * VAL_SPLIT), val_dir)

    print("Starting test data creation...")
    os.makedirs(test_dir, exist_ok=True) if not os.path.exists(test_dir) else None
    create_data(START_INDEX, START_INDEX + int(TRAIN_SIZE * TEST_SPLIT), test_dir)

    # Print run time
    end_time = time.time()
    print("Data generaion took " + str(round(end_time - start_time)) + " seconds")
