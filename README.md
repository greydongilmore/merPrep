
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/trajectoryGuide/merPrep">
    <img src="imgs/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">merPrep</h3>

  <p align="center">
    Microelectrode recording conversion tool to The Brain Imaging Data Structure
    <br />
    <a href="https://github.com/trajectoryGuide/merPrep"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/trajectoryGuide/merPrep/issues">Report Bug</a>
    ·
    <a href="https://github.com/trajectoryGuide/merPrep/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

merPrep is a tool to help export data from common commercial electrophysiology recording systems and store the data according to The Brain Imaging Data Structure. Currently, the tool will read data stored in .txt (from the Leadpoint system) and .mat files (from the AlphaOmega system). 

### Built With

* Python version: 3.9


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* No Prerequisites required. Any Python3 version will suffice.

### Installation

1. In a terminal, clone the repo by running:
    ```sh
    git clone https://github.com/trajectoryGuide/merPrep.git
    ```

2. Change into the project directory (update path to reflect where you stored this project directory):
    ```sh
    cd /home/user/Documents/Github/merPrep
    ```

3. Install the required Python packages:
    ```sh
    python -m pip install -r requirements.txt
    ```


<!-- USAGE EXAMPLES -->
## Usage

1. In a terminal, move into the project directory
     ```sh
     cd /home/user/Documents/Github/merPrep/src
     ```

2. Run the following to execute the epoch script:
    ```sh
    python merPrep.py -i "full/path/to/data/directory"
    ```

  * **-i:** full file path to the raw MER data


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Greydon Gilmore - [@GilmoreGreydon](https://twitter.com/GilmoreGreydon) - greydon.gilmore@gmail.com

Project Link: [https://github.com/trajectoryGuide/merPrep](https://github.com/trajectoryGuide/merPrep)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* README format was adapted from [Best-README-Template](https://github.com/othneildrew/Best-README-Template)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/trajectoryGuide/merPrep.svg?style=for-the-badge
[contributors-url]: https://github.com/trajectoryGuide/merPrep/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/trajectoryGuide/merPrep.svg?style=for-the-badge
[forks-url]: https://github.com/trajectoryGuide/merPrep/network/members
[stars-shield]: https://img.shields.io/github/stars/trajectoryGuide/merPrep.svg?style=for-the-badge
[stars-url]: https://github.com/trajectoryGuide/merPrep/stargazers
[issues-shield]: https://img.shields.io/github/issues/trajectoryGuide/merPrep.svg?style=for-the-badge
[issues-url]: https://github.com/trajectoryGuide/merPrep/issues
[license-shield]: https://img.shields.io/github/license/trajectoryGuide/merPrep.svg?style=for-the-badge
[license-url]: https://github.com/trajectoryGuide/merPrep/blob/master/LICENSE.txt
