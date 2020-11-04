"""
    Social-Distancing

    IIT : Istituto italiano di tecnologia

    Pattern Analysis and Computer Vision (PAVIS) research line

    Description: Social distancing alternative mjpeg reader

    Disclaimer:
    The information and content provided by this application is for information purposes only.
    You hereby agree that you shall not make any health or medical related decision based in whole
    or in part on anything contained within the application without consulting your personal doctor.
    The software is provided "as is", without warranty of any kind, express or implied,
    including but not limited to the warranties of merchantability,
    fitness for a particular purpose and noninfringement. In no event shall the authors,
    PAVIS or IIT be liable for any claim, damages or other liability, whether in an action of contract,
    tort or otherwise, arising from, out of or in connection with the software
    or the use or other dealings in the software.

    LICENSE:
"""

from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

'''
MjpegReader

Read mjpeg multipart streaing file
'''

class MjpegReader:
    def __init__(self, filename):
        try:
            # Open mjpeg file
            self.mjpeg = open(filename, "rb")

            # Initialize TurboJpeg (7x faster than opecv embedded jpeg decoder)
            self.tjpeg = TurboJPEG()

            # File is open
            self.opened = True
        except FileNotFoundError:
            print ("File {0} not found".format(filename))
            self.opened = False

    '''
        Read header line
    '''
    def read_line(self):
        ln = ""
        # Fine end of line
        while True:
            # Read each byte and converto to character
            character = self.mjpeg.read(1).decode("utf-8")

            # Finded end of line, then break loop    
            if character=='\n':
                break
            
            # Concatenate string 
            ln += character

        # Return string
        return ln

    '''
        Read Image
    '''
    def get_image(self):
        # If file is not opened, return None
        if not self.opened:
            return None

        # Read first line (--myboudary) and trash it
        self.read_line()
        
        # Read second line X-Timestamp and store timestamp
        self.ts = int(self.read_line().split(' ')[1])
        
        # Read content type and remove trash it
        self.read_line()

        # Read content lenght and get it to read image
        self.lenght = int(self.read_line().split(' ')[1])

        # Skip blank
        self.read_line()

        # Return decoded image (numpy image format)
        return self.tjpeg.decode(self.mjpeg.read(self.lenght))

    '''
        Get Timestamp
    '''
    def get_ts(self):
        return self.ts

    '''
        Get lenght
    '''
    def get_lenght(self):
        return self.lenght

    '''
        Get file status
    '''
    def is_opened(self):
        return self.opened