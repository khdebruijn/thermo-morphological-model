import subprocess

def start_xbeach(xbeach_path, params_path):
    """
    Running this function starts the XBeach module as a subprocess.
    --------------------------
    xbeach_path: str
        string containing the file path to the xbeach executible from the project directory
    params_path: str
        string containing the file path to the params.txt file from the project directory
    --------------------------

    returns boolean (True if process was a sucess, False if not)
    """

    # Command to run XBeach
    command = [xbeach_path, params_path]

    # Call XBeach using subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and get the return code
    stdout, stderr = process.communicate()
    return_code = process.returncode

    return return_code == 0

def write_xbeach_output(output_path, save_path):
    """
    Running this function writes the xbeach output file (i.e., morphological update) to the wrapper.
    --------------------------
    output_path: str
        string containing the file path to the xbeach output from the project directory
    save_path: str
        string containing the save path for the morphological update
    --------------------------
    """
    
    # Read output file

    # Convert to correct format

    # Save output file

    return 