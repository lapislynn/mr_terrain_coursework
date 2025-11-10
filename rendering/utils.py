import glm
from math import radians, tan


def sign (p1, p2, p3):
	'''
	Function to determine the sign of a point relative to a line formed by two other points
	'''
	return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def PointInTriangle (pt, v1, v2, v3):
	'''
	Function to check if a point is inside a triangle formed by three other points
	'''
	d1 = sign(pt, v1, v2)
	d2 = sign(pt, v2, v3)
	d3 = sign(pt, v3, v1)
	has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
	has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
	return not (has_neg and has_pos)


def giveTriangleVertices(pos, front, fov, far):
	'''
	Function to calculate vertices of a triangle based on position, view direction, field of view, and distance
	'''
	front = glm.normalize(front)
	perp  = glm.vec2(front[1], -front[0])

	mag_b = far + far/4
	mag_p = mag_b * tan(radians(fov/1.5))

	v1 = pos - front * 1.25
	v2 = pos + mag_b * front + mag_p * perp
	v3 = pos + mag_b * front - mag_p * perp

	return v1, v2, v3


