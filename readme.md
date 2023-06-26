# Segmented Cavity Detection

   This project aims to showcase cavity detection on segmented periapical dental radiographs.

## Prerequisites

Before building and executing the Docker image, make sure you have the following prerequisites installed on your machine:

- Docker: [Install Docker](https://docs.docker.com/get-docker/)
- Git: [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## Build and Execute the Docker Image

To build and execute the Docker image for the Segmented Cavity Detection project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/nitou2504/segmented-cavity-detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd segmented-cavity-detection
   ```

3. Build the Docker image:

   ```bash
   docker build -t segmented-cavity-detection .
   ```

   This may take some time as it installs the necessary dependencies.

## Run the Docker Image

To run the Docker image for the Segmented Cavity Detection project, follow these steps:

1. Create a folder named `raw` in the base directory of the project.

2. Inside the `raw` folder, place the periapical dental radiographs in JPG format. Each image should be accompanied by an XML file with the same name. The XML file should be formatted as Pascal VOC XML, providing information about the objects in the image, using the classes `caries` and `no_caries`.

   The folder structure and file format should look like this:

   ```
   /path/to/base/dir
   └── raw
       ├── image1.jpg
       ├── image1.xml
       ├── image2.jpg
       ├── image2.xml
       └── ...
   ```

   Note: The XML files should follow the Pascal VOC XML format, which includes annotations for the objects in the image, such as bounding boxes and class labels.

3. Run the Docker image using the following command:

   ```bash
   docker run -it -e BASE_DIR=. -v .:/app segmented-cavity-detection
   ```

   This command executes the Docker container in interactive mode inside the project directory  `segmented-cavity-detection`. Inside the container, the base directory will be available as the `$BASE_DIR` environment variable, and you will be in the container's shell.

   Make sure to run the docker image inside the project directory, so it can access the python scripts and image files.

   Note: The `--entrypoint=/bin/bash` flag can be used to start an interactive shell instead of running the default entrypoint specified in the Docker image, that runs the whole experiment starting from preprosessing the raw images. You can exit the container by typing `exit` in the shell.

## License

This project is licensed under the GNU General Public License (GPL) version 3.0. See the [LICENSE](LICENSE) file for more details.
