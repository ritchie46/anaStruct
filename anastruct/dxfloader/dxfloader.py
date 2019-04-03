import os
import logging

import ezdxf

from anastruct.dxfloader.geo_utils import Point, Line

def round_coordinate(point, dig=5):
    point[0] = round(point[0], dig)
    point[1] = round(point[1], dig)

element_commands = ['add_element', 'add_multiple_elements', 'add_truss_element']

point_commands = ['add_support', 'point_load', 'moment_load']
    
class DxfLoader():

    def __init__(self):
        self.dxf_file_path = None
        self.dwg = None
        self.system=None
        #--dxf content
        self.DXF_LINES = []
        self.DXF_TEXTS_ELEMENT_COMMAND = []
        self.DXF_TEXTS_POINT_COMMAND = []

    def load_dxf_data_to_system(self, dxf_file_path, system):
        self.system=system
        #---
        if os.path.isfile(dxf_file_path):
            self._open_dxf(dxf_file_path)
            self._import_dxf_content()
            self._execute_command_from_dxf()
        else:
            logging.error('File %s does not exist!!'%dxf_file_path)
        #---
        self.system = None

    def _open_dxf(self, dxf_file_path):
        self.dxf_file_path = dxf_file_path
        self.dwg = ezdxf.readfile(self.dxf_file_path)

    def _import_dxf_content(self):
        #---
        LINES = []
        for e in self.dwg.modelspace():
            if e.dxftype() == 'LINE':
                LINES.append(e)
        self.DXF_LINES = LINES
        #---
        TEXTS = []
        for e in self.dwg.modelspace():
            if e.dxftype() == 'TEXT':
                if any([i in e.dxf.text for i in element_commands]):
                    TEXTS.append(e)
        self.DXF_TEXTS_ELEMENT_COMMAND = TEXTS
        #---
        TEXTS = []
        for e in self.dwg.modelspace():
            if e.dxftype() == 'TEXT':
                if any([i in e.dxf.text for i in point_commands]):
                    TEXTS.append(e)
        self.DXF_TEXTS_POINT_COMMAND = TEXTS

    def _execute_command_from_dxf(self):
        #---executing element commands
        for line in self.DXF_LINES:
            line_start_point = list(line.dxf.start[:2])
            round_coordinate(line_start_point)
            line_end_point = list(line.dxf.end[:2])
            round_coordinate(line_end_point)
            #--
            dxf_command_text = self._get_nearest_text_to_line(Line(Point(line_start_point), Point(line_end_point)))
            if dxf_command_text:
                command = 'self.system.' + dxf_command_text.replace('(', '(location=' + str([line_start_point, line_end_point]) + ',')
                exec(command)
            else:
                self.system.add_element([line_start_point, line_end_point])
        #---executing point commands
        for point in self._get_model_points():
            dxf_command_text = self._get_nearest_text_to_point(Point(point))
            if dxf_command_text:
                id = self.system.find_node_id(point)
                command = 'self.system.' + dxf_command_text.replace('(', '(node_id=' + str(id) + ',')
                exec(command)

    def _get_nearest_text_to_line(self, line):
        if not self.DXF_TEXTS_ELEMENT_COMMAND:
            return None
        tdists = []
        for txt in self.DXF_TEXTS_ELEMENT_COMMAND:
            dxfinsert = txt.dxf.insert
            textp = Point([dxfinsert[0], dxfinsert[1]])
            txtdist = line.distance(textp)
            tdists.append(txtdist)
        mintdist = min(tdists)
        mintdistindex = tdists.index(mintdist)
        t_entity = self.DXF_TEXTS_ELEMENT_COMMAND[mintdistindex] #the closest text entity
        maxdist = 2.0 * t_entity.dxf.height
        #---and if the text is closer than it 2xheight it is the one we looking for
        if mintdist < maxdist:
            return t_entity.dxf.text
        else:
            return None

    def _get_nearest_text_to_point(self, point):
        if not self.DXF_TEXTS_POINT_COMMAND:
            return None
        tdists = []
        for txt in self.DXF_TEXTS_POINT_COMMAND:
            dxfinsert = txt.dxf.insert
            textp = Point([dxfinsert[0], dxfinsert[1]])
            txtdist = point.distance(textp)
            tdists.append(txtdist)
        mintdist = min(tdists)
        mintdistindex = tdists.index(mintdist)
        t_entity = self.DXF_TEXTS_POINT_COMMAND[mintdistindex] #the closest text entity
        maxdist = 2.0 * t_entity.dxf.height
        #---and if the text is closer than it 2xheight it is the one we looking for
        if mintdist < maxdist:
            return t_entity.dxf.text
        else:
            return None
    
    def _get_model_points(self):
        points = []
        for line in self.DXF_LINES:
            line_start_point = list(line.dxf.start[:2])
            round_coordinate(line_start_point)
            line_end_point = list(line.dxf.end[:2])
            round_coordinate(line_end_point)
            for point in [line_start_point, line_end_point]:
                if not point in points: points.append(point)
        return points

dxfloader = DxfLoader()