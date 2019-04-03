import os

from anastruct.fem.system import SystemElements

THIS_SCRIPT_DIR_PATH = (os.path.dirname(os.path.abspath(__file__)))
DXF_FILE_NAME = 'ex_24_model.dxf'
DXF_FILE_PATH = os.path.join(THIS_SCRIPT_DIR_PATH, DXF_FILE_NAME)

ss = SystemElements()

ss.load_dxf_data_to_system(DXF_FILE_PATH)

ss.show_structure(annotations=False)
ss.solve()
ss.show_displacement()