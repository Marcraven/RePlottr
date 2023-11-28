import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import random
import os
import sys


### Define constants
train_size = 800
val_split = 0.125
test_split = 0.125

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

colors = [
    "b",
    "g",
    "r",
    "c",
    "m",
    "y",
    "k",  # Single-letter abbreviations for basic colors
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "white",  # Full color names
    "skyblue",
    "tomato",
    "gold",
    "purple",
    "lime",
    "orange",
    "pink",
    "brown",  # Some named colors
    "#FF5733",
    "#33FF57",
    "#5733FF",  # Hexadecimal color values
    (0.1, 0.2, 0.3),
    (0.4, 0.5, 0.6),
    (0.7, 0.8, 0.9),  # RGB tuples
]


### Create data function
def create_data(start, end, folder):
    """This generates a number of Train, Evaluate or Test data on a specific folder
    start: number of first scatterplot picture
    end: number of last scatterplot picture
    folder: location for saving final pictures
    """

    metadata_list = []

    for j in range(start, end):
        # Generate random data
        xlim = np.random.randint(low=0, high=1000, size=1)
        ylim = np.random.randint(low=0, high=1000, size=1)
        num_series = 1  # np.random.randint(1, 1)
        num_points = np.random.randint(10, 40)

        # Create an empty list to store series data
        series = []

        # Generate and plot random data for each series
        fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100)

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
                color=random.choice(colors),
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

        # Create file names
        fname = folder + str(j).zfill(4)

        # Save the plot with smaller margins
        fig.savefig(
            fname + ".jpg",
            dpi=100,
            bbox_inches=Bbox.from_bounds(-0.26, -0.2, 3.2, 2.56),
        )

        ##### Define Yolo target #####
        yolo_target = np.empty((0, 5))

        ##### Add position coordinates of x axis ticks and y axis ticks to Yolo putput #####

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
        x_ticks_y_coord_std = x_ticks_y_coord_pixel_flip / fig_height

        y_ticks_x_coord_std = y_ticks_x_coord_pixel / fig_width
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


if __name__ == "__main__":
    plt.ioff()

    ### Here we define the size of the dataset and the splits
    if len(sys.argv) > 1:
        train_size = int(sys.argv[1])

    dataset = "ObjectRecognition/yolo/dataset"  # f'./Dataset_{train_size}_' + str(val_split).replace(".", "") +'_' + str(test_split).replace(".", "")
    train_dir = dataset + "/train/"
    val_dir = dataset + "/validation/"
    test_dir = dataset + "/test/"

    print("Starting training data creation...")
    os.makedirs(train_dir, exist_ok=True) if not os.path.exists(train_dir) else None
    create_data(0, train_size, train_dir)
    print("Starting evaluation data creation...")
    os.makedirs(val_dir, exist_ok=True) if not os.path.exists(val_dir) else None
    create_data(0, int(train_size * val_split), val_dir)
    print("Starting test data creation...")
    os.makedirs(test_dir, exist_ok=True) if not os.path.exists(test_dir) else None
    create_data(0, int(train_size * test_split), test_dir)
