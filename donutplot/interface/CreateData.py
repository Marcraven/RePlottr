import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import sys
import json
import yaml
import time
import multiprocessing
from donutplot.params import *

##### Define constants #####
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

figsize_widths = [3.2, 4.8, 6.4]
figsize_heights = [2.4, 3.2, 4.8]
background_colors = ["white", "#F5F5F5", "#D3D3D3"]
fig_dpis = [100, 200, 300]

font_types = ["DejaVu Sans", "sans-serif", "serif"]
font_styles = ["normal", "italic", "oblique"]
font_weights = ["normal", "bold", "light"]

plotting_tools = ["plt", "sns"]

alpha_min = 0.7
alpha_max = 0.9


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

        # Define text properties
        font_type = random.choice(font_types)

        if TRAINING_MODE:
            font_style = "normal"
            font_weight = "normal"
        else:
            font_style = random.choice(font_styles)
            font_weight = random.choice(font_weights)

        font = {"family": font_type, "style": font_style, "weight": font_weight}
        mpl.rc("font", **font)

        # Define figure size, background colour and dpi
        background_color = random.choice(background_colors)

        if TRAINING_MODE:
            figsize_width = FIGSIZE_WIDTH_TRAINING_MODE
            figsize_height = FIGSIZE_HEIGHT_TRAINING_MODE
            dpi = DPI_TRAINING_MODE

        else:
            figsize_width = random.choice(figsize_widths)
            figsize_height = random.choice(figsize_heights)
            dpi = random.choice(fig_dpis)

        # Define figure and ax
        fig, ax = plt.subplots(
            figsize=(figsize_width, figsize_height),
            facecolor=background_color,
            dpi=dpi,
        )

        # Define chart plotting tool
        plotting_tool = random.choice(plotting_tools)

        # Define alpha
        alpha_switch = random.choice([False, True])
        alpha = round(np.random.uniform(alpha_min, alpha_max), 1) if alpha_switch else 1

        # Create an empty list to store series data
        series = []

        # Generate and plot random data for each series
        for i in range(num_series):
            # Create series
            name = random.choice(legends)
            points_serie = max(num_points // num_series + np.random.randint(-2, 2), 1)
            x_data = np.round(np.random.rand(points_serie) * xlim, decimals=1)
            y_data = np.round(np.random.rand(points_serie) * ylim, decimals=1)
            marker = random.choice(markers)
            dot_color = random.choice(dot_colors)
            series.append(
                {
                    "marker": marker,
                    "x_values": list(x_data),
                    "y_values": list(y_data),
                }  # "name": name,
            )

            # Create a scatter plot for the current series
            if plotting_tool == "plt":
                ax.scatter(
                    x=x_data,
                    y=y_data,
                    label=name,
                    marker=marker,
                    color=dot_color,
                    alpha=alpha,
                )

            elif plotting_tool == "sns":
                sns.scatterplot(
                    x=x_data,
                    y=y_data,
                    ax=ax,
                    label=name,
                    color=dot_color,
                    marker=marker,
                    legend=False,
                    alpha=alpha,
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

        # Decide whether tickmarks are shown
        tickmark_remove_switch = random.choices([False, True], weights=[0.7, 0.3])[0]

        if tickmark_remove_switch:
            ax.tick_params(axis="both", which="both", length=0)

        # Decide whether top and right spines are shown
        spine_remove_switch = random.choice([False, True])

        if spine_remove_switch:
            ax.spines[["right", "top"]].set_visible(False)

        # Decide whether gridlines are shown
        gridline_switch = random.choices([False, True], weights=[0.7, 0.3])[0]
        x_grid_switch = (
            random.choices([False, True], weights=[0.2, 0.8])[0]
            if gridline_switch
            else False
        )
        y_grid_switch = (
            random.choices([False, True], weights=[0.2, 0.8])[0]
            if gridline_switch
            else False
        )

        if x_grid_switch and y_grid_switch:
            ax.grid(linewidth=0.5)
        elif x_grid_switch and not y_grid_switch:
            ax.grid(axis="x", linewidth=0.5)
        elif not x_grid_switch and y_grid_switch:
            ax.grid(axis="y", linewidth=0.5)

        # Set tight layout
        fig.tight_layout()

        # Create file name
        fname = os.path.join(folder, f"{str(j).zfill(4)}.jpg")

        # Save the plot with smaller margins
        fig.savefig(
            fname,
            dpi=dpi,
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
            "x_label": [
                [
                    x_label.split("[")[0].rstrip(" "),
                    x_label.split("[")[1].rstrip("]"),
                ]
            ],
            "y_label": [
                [
                    y_label.split("[")[0].rstrip(" "),
                    y_label.split("[")[1].rstrip("]"),
                ]
            ],
            "data_dicts": series,
            # "x_ticks": list(x_ticks_data),
            # "y_ticks": list(y_ticks_data),
        }

        # Create metadata
        metadata = {
            "file_name": str(j).zfill(4) + ".jpg",
            "ground_truth": ground_truth,
        }
        metadata_list.append(metadata)

        # Define Yolo target
        yolo_target = np.empty((0, 5))

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
            x_ticks_y_coord_pixel_flip / fig_height
            + 0.05 * 2.4 / figsize_height  # Note manual adjustment
        )

        y_ticks_x_coord_std = (
            y_ticks_x_coord_pixel / fig_width
            - 0.075 * 3.2 / figsize_width  # Note manual adjustment
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
                        0.15 * 3.2 / figsize_width * np.ones((len(x_ticks_data), 1)),
                        0.1 * 2.4 / figsize_height * np.ones((len(x_ticks_data), 1)),
                    )
                ),
                np.hstack(
                    (
                        1 * np.ones((len(y_ticks_data), 1)),
                        np.expand_dims(y_ticks_x_coord_std, 1),
                        np.expand_dims(y_ticks_y_coord_std, 1),
                        0.15 * 3.2 / figsize_width * np.ones((len(y_ticks_data), 1)),
                        0.1 * 2.4 / figsize_height * np.ones((len(y_ticks_data), 1)),
                    )
                ),
            )
        )

        ##### Add dots' position coordinates to Yolo output #####
        for i in range(num_series):
            ### Obtain series data
            x_data = series[i]["x_values"]
            y_data = series[i]["y_values"]
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
                            0.06 * 3.2 / figsize_width * np.ones((len_xy, 1)),
                            0.06 * 2.4 / figsize_height * np.ones((len_xy, 1)),
                        )
                    ),
                )
            )

        # Save Yolo target in txt format that is read by Yolo model
        np.savetxt(
            os.path.join(folder, f"{str(j).zfill(4)}.txt"),
            yolo_target,
            delimiter=" ",
            fmt="%1.4f",
        )

        # Clean figure, axes and close figure
        plt.clf()
        plt.cla()
        plt.close()

    # Generate JSONL file
    file_path = os.path.join(folder, "metadata.jsonl")

    with open(file_path, "w") as file:
        for item in metadata_list:
            json.dump(item, file, default=str)  # Use str() for non-serializable objects
            file.write("\n")  # Add a newline character to separate JSON objects

    # Generate YAML file
    yaml_path = os.path.join(DATA_PATH, "dataset.yaml")
    lines = [
        "train: train/",
        "val: validation/",
        "test: test/",
        "names:",
        f"\t0: x-ticks",
        f"\t1: y-ticks",
        f'\t2: "."',
        f'\t3: "o"',
        f'\t4: "v"',
        f'\t5: "^"',
        f'\t6: "<"',
        f'\t7: ">"',
        f'\t8: "1"',
        f'\t9: "2"',
        f'\t10: "3"',
        f'\t11: "4"',
        f'\t12: "s"',
        f'\t13: "p"',
        f'\t14: "*"',
        f'\t15: "h"',
        f'\t16: "H"',
        f'\t17: "+"',
        f'\t18: "x"',
        f'\t19: "D"',
        f'\t20: "d"',
    ]

    with open(yaml_path, "w") as file:
        for each in lines:
            file.writelines(
                f"\n{each}"
            )  # Add a newline character to separate YAML lines


##### Define data generation steps #####
def generate_data(data_type, start_index, size, directory):
    print(f"Starting {data_type} data creation...")
    os.makedirs(directory, exist_ok=True)
    create_data(start_index, start_index + size, directory)


##### If name = main #####
if __name__ == "__main__":
    # Record start time
    start_time = time.time()

    # Calculate CPU cores
    num_cores = multiprocessing.cpu_count()

    # Turn interactive mode off
    plt.ioff()

    # Define arguments and folders
    if len(sys.argv) > 1:
        TRAIN_SIZE = int(sys.argv[1])

    # Define tasks
    tasks = [
        ("train", START_INDEX, TRAIN_SIZE, TRAIN_PATH),
        ("validation", START_INDEX, int(TRAIN_SIZE * VAL_SPLIT), VALIDATE_PATH),
        ("test", START_INDEX, int(TRAIN_SIZE * TEST_SPLIT), TEST_PATH),
    ]

    # Initiate multiprocessing
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.starmap(generate_data, tasks)

    # Print run time
    end_time = time.time()
    print("Data generaion took " + str(round(end_time - start_time)) + " seconds")
