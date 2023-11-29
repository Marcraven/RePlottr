import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import random
import os
import sys


### Here we define the constants
train_size = 100
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
    ",",
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
    """This generates a number of Train, Evaluate or Test data on a specific folder"""

    metadata_list = []
    for j in range(start, end):
        # Generate random data
        xlim = np.random.randint(low=0, high=1000, size=1)
        ylim = np.random.randint(low=0, high=1000, size=1)
        num_series = 1  # np.random.randint(1, 1)
        num_points = np.random.randint(10, 40)

        # Create an empty list to store series data
        series = []

        plt.figure(figsize=(3.2, 2.4), dpi=100)
        # Generate and plot random data for each series
        for i in range(num_series + 1):
            name = random.choice(legends)
            points_serie = max(num_points // num_series + np.random.randint(-2, 2), 1)
            x_values = np.round(np.random.rand(points_serie) * xlim, decimals=1)
            y_values = np.round(np.random.rand(points_serie) * ylim, decimals=1)
            series.append(name)  # 'x': list(x_values), 'y': list(y_values)})

            # Create a scatter plot for the current series
            plt.scatter(
                x=x_values,
                y=y_values,
                label=name,
                marker=random.choice(markers),
                color=random.choice(colors),
            )
        plt.legend(loc="upper right", framealpha=0.3)  # , bbox_to_anchor=(0.6,0.5))
        # Add labels and title
        x_label = random.choice(labels)
        y_label = random.choice(labels)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plot_title = random.choice(adjectives) + " " + random.choice(nouns)
        plt.title(plot_title)

        x_ticks = plt.xticks()[0].tolist()
        y_ticks = plt.xticks()[0].tolist()

        # file names
        fname = folder + str(j).zfill(4)
        # Save the plot with smaller margins
        plt.savefig(
            fname + ".jpg",
            dpi=100,
            bbox_inches=Bbox.from_bounds(-0.26, -0.2, 3.2, 2.56),
        )

        # Create ground truth dictionary for DONUT and add it to metadata

        ground_truth = {
            "title": plot_title,
            "x_label": x_label,
            "x_ticks": list(plt.xticks()[0].tolist()),
            "y_label": y_label,
            "y_ticks": list(y_ticks),
            "series": series,
        }
        metadata = {
            "file_name": str(j).zfill(4) + ".jpg",
            "ground_truth": '{"gt_parse": ' + json.dumps(ground_truth) + "}",
        }
        metadata_list.append(metadata)

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
    plt.ioff()

    ### Here we define the size of the dataset and the splits
    if len(sys.argv) > 1:
        train_size = int(sys.argv[1])

    dataset = "./TextRecognition/DonutApproach/dataset"  # f'./Dataset_{train_size}_' + str(val_split).replace(".", "") +'_' + str(test_split).replace(".", "")
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
