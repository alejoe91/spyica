#!/usr/bin/env python
''' Clean up!!! '''


import os
from os.path import join
import numpy as np
import math

# import matplotlib as 'Agg' to avoid interference with neuron
import matplotlib
# matplotlib.use('Agg')
import pylab as plt
# import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
import LFPy
import MEAutility as MEA
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection
from matplotlib import colors as mpl_colors
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# plt.ion()

# TODO update plots to cope with different MEAs


def _rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis /= np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def _cylinder(pos_start,
              direction,
              length,
              radius,
              n_points,
              flatten_along_zaxis=False):
    alpha = np.array([0., length])

    theta_ring = np.linspace(0., np.pi * 2., n_points)
    r = radius

    x = np.zeros((theta_ring.size * alpha.size))
    y = np.zeros((theta_ring.size * alpha.size))
    z = np.zeros((theta_ring.size * alpha.size))

    for idx_alpha, crt_alpha in enumerate(alpha):
        x[idx_alpha * theta_ring.size:
        (idx_alpha + 1) * theta_ring.size] = \
            r * np.cos(theta_ring)
        y[idx_alpha * theta_ring.size:
        (idx_alpha + 1) * theta_ring.size] = \
            r * np.sin(theta_ring)
        z[idx_alpha * theta_ring.size:
        (idx_alpha + 1) * theta_ring.size] = \
            crt_alpha * np.ones(theta_ring.size)

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)

    d = direction

    # rot1, phi
    r_1 = np.array([0., 1., 0.])
    # rot2, theta
    r_2 = np.array([0., 0., 1.])

    # fix negative angles
    if d[0] == 0:
        theta = -np.sign(d[1])*np.pi / 2.
    else:
        if d[0] > 0:
            theta = -np.arctan(d[1] / d[0])
        else:
            theta = np.pi - np.arctan(d[1] / d[0])


    # print 'theta: ', np.rad2deg(theta)

    rho = np.sqrt((d[0] ** 2 + d[1] ** 2))

    if rho == 0:
        phi = 0.
    else:
        phi = -(np.pi / 2. - np.arctan(d[2] / rho))

    # print 'phi: ', np.rad2deg(phi)

    rot1_m = _rotation_matrix(r_1, phi)
    rot2_m = _rotation_matrix(r_2, theta)

    for idx, (crt_x, crt_y, crt_z) in enumerate(zip(x[0], y[0], z[0])):
        crt_v = np.array([crt_x, crt_y, crt_z])
        crt_v = np.dot(crt_v, rot1_m)
        crt_v = np.dot(crt_v, rot2_m)
        x[0][idx] = crt_v[0]
        y[0][idx] = crt_v[1]
        z[0][idx] = crt_v[2]

    x += pos_start[0]
    y += pos_start[1]
    z += pos_start[2]
    if flatten_along_zaxis is True:
        z = np.abs(z)
        z *= 0.00000000001
    return x, y, z


def get_polygons_for_cylinder(pos_start,
                              direction,
                              length,
                              radius,
                              n_points,
                              facecolor='b',
                              edgecolor='k',
                              alpha=1.,
                              lw = 0.,
                              flatten_along_zaxis=False):
    x, y, z = _cylinder(pos_start,
                        direction,
                        length,
                        radius,
                        n_points,
                        flatten_along_zaxis)

    alpha_tup = alpha,
    edge_col = mpl_colors.to_rgb(edgecolor) + alpha_tup
    face_col = mpl_colors.to_rgb(facecolor) + alpha_tup

    theta_ring = np.linspace(0., np.pi * 2., n_points)
    verts_hull = []
    for idx_theta, crt_theta in enumerate(theta_ring):
        if idx_theta <= theta_ring.size - 2:
            x_verts = [x[0][idx_theta],
                       x[0][idx_theta + 1],
                       x[0][idx_theta + 1 + theta_ring.size],
                       x[0][idx_theta + theta_ring.size]]
            y_verts = [y[0][idx_theta],
                       y[0][idx_theta + 1],
                       y[0][idx_theta + 1 + theta_ring.size],
                       y[0][idx_theta + theta_ring.size]]
            z_verts = [z[0][idx_theta],
                       z[0][idx_theta + 1],
                       z[0][idx_theta + 1 + theta_ring.size],
                       z[0][idx_theta + theta_ring.size]]
            verts_hull.append([zip(x_verts, y_verts, z_verts)])

    poly3d_hull = []
    for crt_vert in verts_hull:
        cyl = Poly3DCollection(crt_vert, linewidths=lw)
        cyl.set_facecolor(face_col)
        cyl.set_edgecolor(edge_col)

        poly3d_hull.append(cyl)

    # draw lower lid
    x_verts = x[0][0:theta_ring.size - 1]
    y_verts = y[0][0:theta_ring.size - 1]
    z_verts = z[0][0:theta_ring.size - 1]
    verts_lowerlid = [zip(x_verts, y_verts, z_verts)]
    poly3ed_lowerlid = Poly3DCollection(verts_lowerlid, linewidths=lw, zorder=1)
    poly3ed_lowerlid.set_facecolor(face_col)
    poly3ed_lowerlid.set_edgecolor(edge_col)

    # draw upper lid
    x_verts = x[0][theta_ring.size:theta_ring.size * 2 - 1]
    y_verts = y[0][theta_ring.size:theta_ring.size * 2 - 1]
    z_verts = z[0][theta_ring.size:theta_ring.size * 2 - 1]
    verts_upperlid = [zip(x_verts, y_verts, z_verts)]
    poly3ed_upperlid = Poly3DCollection(verts_upperlid, linewidths=lw, zorder=1)
    poly3ed_upperlid.set_facecolor(face_col)
    poly3ed_upperlid.set_edgecolor(edge_col)

    return_col = poly3d_hull
    return_col.append(poly3ed_lowerlid)
    return_col.append(poly3ed_upperlid)
    return return_col


def plot_detailed_neuron(cell=None, cell_name=None, cell_folder=None, pos=None, rot=None, plane=None, ax=None,
                         bounds=None, alpha=None, color=None, c_axon=None, c_dend=None, c_soma=None, plot_axon=True,
                         xlim=None, ylim=None):


    if cell is None:
        if cell_name is None or cell_folder is None:
            raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')
        folder = join(cell_folder, cell_name)
        cwd = os.getcwd()
        os.chdir(folder)
        morphologyfile = os.listdir('morphology')[0]  # glob('morphology\\*')[0]
        cell = LFPy.Cell(morphology=join('morphology', morphologyfile),
                         pt3d=True)
        os.chdir(cwd)
    elif type(cell) is not LFPy.TemplateCell and type(cell) is not LFPy.Cell:
        raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')

    if pos is not None:
        if len(pos) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_pos(pos[0], pos[1], pos[2])
    if rot is not None:
        if len(rot) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_rotation(rot[0], rot[1], rot[2])

    if alpha:
        alp = alpha
    else:
        alp = 0.8
    if color:
        col = color
    else:
        col = 'gray'
    if c_axon:
        col_ax = c_axon
    else:
        col_ax = col
    if c_dend:
        col_dend = c_dend
    else:
        col_dend = col
    if c_soma:
        col_soma = c_soma
    else:
        col_soma = col
    if ax is None:
        fig = plt.figure()
        if plane is not '3d':
            axis = fig.add_subplot(111)
        else:
            axis = fig.add_subplot(111, projection='3d')
    else:
        axis = ax

    idx_ax = cell.get_idx('axon')
    idx_dend = cell.get_idx('dend')
    idx_apic = cell.get_idx('apic')
    idx_soma = cell.get_idx('soma')

    idx_ax3d = []
    idx_dend3d = []
    idx_apic3d = []
    idx_soma3d = []

    for i in range(len(cell.x3d)):
        mid = len(cell.x3d[i]) // 2
        idx = cell.get_closest_idx(cell.x3d[i][mid],cell.y3d[i][mid], cell.z3d[i][mid])
        if idx in idx_ax:
            idx_ax3d.append(i)
        elif idx in idx_dend:
            idx_dend3d.append(i)
        elif idx in idx_apic:
            idx_apic3d.append(i)
        elif idx in idx_soma:
            idx_soma3d.append(i)

    zips = []

    if plane is '3d':
        '''USE x3d, y3d, z3d and nested cycle...not better'''

        for ii in range(len(cell.x3d)):
            print 'seg: [', ii, '/', len(cell.x3d), ']'
            for jj in range(len(cell.x3d[ii])-1):
                init = np.array([cell.x3d[ii][jj], cell.y3d[ii][jj], cell.z3d[ii][jj]])
                end = np.array([cell.x3d[ii][jj+1], cell.y3d[ii][jj+1], cell.z3d[ii][jj+1]])
                dir_seg = (end -init) / np.linalg.norm(end-init)
                len_seg = np.linalg.norm(end-init)
                n_points = 5.
                neur_poly3d = get_polygons_for_cylinder(init,
                                                         direction=dir_seg,
                                                         length=len_seg,
                                                         radius=cell.diam3d[ii][jj]/2,
                                                         n_points=n_points,
                                                         facecolor=col,
                                                         alpha=alp,
                                                         flatten_along_zaxis=False)
                for crt_poly3d in neur_poly3d:
                    axis.add_collection3d(crt_poly3d)

        axis.set_xlim3d(np.min(cell.xmid), np.max(cell.xmid))
        axis.set_ylim3d(np.min(cell.ymid), np.max(cell.ymid))
        axis.set_zlim3d(np.min(cell.zmid), np.min(cell.zmid))

    elif plane is 'yz' or plane is 'zy':
        for y, z in cell.get_pt3d_polygons(projection=('y', 'z')):
            zips.append(zip(y, z))

    elif plane is 'xz' or plane is 'zx':
        for x, z in cell.get_pt3d_polygons(projection=('x', 'z')):
            zips.append(zip(x, z))

    elif plane is 'xy' or plane is 'yx':
        for x, y in cell.get_pt3d_polygons(projection=('x', 'y')):
            zips.append(zip(x, y))

    if plane is not '3d':
        axon_pol = [zips[i] for i in idx_ax3d]
        dend_pol = [zips[i] for i in idx_dend3d]
        apic_pol = [zips[i] for i in idx_apic3d]
        soma_pol = [zips[i] for i in idx_soma3d]
        if plot_axon:
            polycol_ax = PolyCollection(axon_pol,
                                     edgecolors='none',
                                     facecolors=col_ax,
                                     alpha=alp)
            axis.add_collection(polycol_ax)
        polycol_dend = PolyCollection(dend_pol,
                                 edgecolors='none',
                                 facecolors=col_dend,
                                 alpha=alp)
        polycol_apic = PolyCollection(apic_pol,
                                 edgecolors='none',
                                 facecolors=col_dend,
                                 alpha=alp)
        polycol_soma = PolyCollection(soma_pol,
                                 edgecolors='none',
                                 facecolors=col_soma,
                                 alpha=alp)


        axis.add_collection(polycol_dend)
        axis.add_collection(polycol_apic)
        axis.add_collection(polycol_soma)    

    if bounds is not None:
        if len(bounds) == 4:
            axis.axis('on')
            axis.set_xlim([bounds[0], bounds[1]])
            axis.set_ylim([bounds[2], bounds[3]])
            axis.axis('off')
    else:
        if plane is '3d':
            axis.set_xlim3d(np.min(cell.xmid), np.max(cell.xmid))
            axis.set_ylim3d(np.min(cell.ymid), np.max(cell.ymid))
            axis.set_zlim3d(np.min(cell.zmid), np.max(cell.zmid))
        else:
            if xlim:
                axis.set_xlim(xlim)
            if ylim:
                axis.set_ylim(ylim)
            if not xlim and not ylim:
                axis.axis('equal')

    return axis


def plot_neuron(cell=None, cell_name=None, cell_folder=None, pos=None, rot=None, bounds=None, plane=None,
                fig=None,ax=None, projections3d=False, alpha=None, color=None, condition=None,
                c_axon=None, c_dend=None, c_soma=None, plot_axon=True, plot_dend=True, plot_soma=True,
                xlim=None, ylim=None, somasize=30):

    # in LFPy you can get all points of tue neuron
    # cell.get
    if cell is None:
        if cell_name is None or cell_folder is None:
            raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')
        folder = join(cell_folder, cell_name)
        cwd = os.getcwd()
        print folder
        os.chdir(folder)
        morphologyfile = os.listdir('morphology')[0]  # glob('morphology\\*')[0]
        print join('morphology', morphologyfile)
        cell = LFPy.Cell(morphology=join('morphology', morphologyfile))
        os.chdir(cwd)
    elif type(cell) is not LFPy.TemplateCell and type(cell) is not LFPy.Cell:
        raise AttributeError('Either a Cell object or the cell name and location should be passed as parameters')

    # cell = return_cell_shape(folder, cell_name)
    if pos is not None:
        if len(pos) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_pos(pos[0], pos[1], pos[2])
    if rot is not None:
        if len(rot) != 3:
            print 'Input a single posiion at a time'
        else:
            cell.set_rotation(rot[0], rot[1], rot[2])

    if plane is None:
        plane = 'yz'
    if alpha:
        alp = alpha
    else:
        alp = 0.8
    if color:
        col = color
    else:
        col = 'gray'
    if c_axon:
        col_ax = c_axon
    else:
        col_ax = col
    if c_dend:
        col_dend = c_dend
    else:
        col_dend = col
    if c_soma:
        col_soma = c_soma
    else:
        col_soma = col

    idx_ax = cell.get_idx('axon')
    idx_dend = cell.get_idx('dend')
    idx_soma = cell.get_idx('soma')

    if projections3d:
        if fig is None:
            fig = plt.figure()
        yz = fig.add_subplot(221, aspect=1)
        xy = fig.add_subplot(222, aspect=1)
        xz = fig.add_subplot(223, aspect=1)
        threeD = fig.add_subplot(224, projection='3d')

        for idx in range(cell.totnsegs):
            if idx == cell.somaidx:
                yz.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], 'r', lw=5)
            elif idx in idx_ax:
                yz.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], 'b')
            else:
                yz.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], 'k')
        for idx in range(cell.totnsegs):
            if idx == cell.somaidx:
                xy.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], 'r', lw=5)
            elif idx in idx_ax:
                xy.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], 'b')
            else:
                xy.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], 'k')
        for idx in range(cell.totnsegs):
            if idx == cell.somaidx:
                xz.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], 'r', lw=5)
            elif idx in idx_ax:
                xz.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], 'b')
            else:
                xz.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], 'k')

        for idx in range(cell.totnsegs):
            if idx == cell.somaidx:
                threeD.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                            [cell.zstart[idx], cell.zend[idx]], 'r', lw=5)
            elif idx in idx_ax:
                threeD.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                            [cell.zstart[idx], cell.zend[idx]], 'b')
            else:
                threeD.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                            [cell.zstart[idx], cell.zend[idx]], 'k')

        yz.set_xlabel('y ($\mu$m)')
        yz.set_ylabel('z ($\mu$m)')
        xy.set_xlabel('x ($\mu$m)')
        xy.set_ylabel('y ($\mu$m)')
        xz.set_xlabel('x ($\mu$m)')
        xz.set_ylabel('z ($\mu$m)')
        threeD.set_xlabel('x ($\mu$m)')
        threeD.set_ylabel('y ($\mu$m)')
        threeD.set_zlabel('z ($\mu$m)')

        return fig 

    else:

        if ax:
            axis = ax
        else:
            fig = plt.figure()
            if plane is not '3d':
                axis = fig.add_subplot(111, aspect=1)
            else:
                axis = fig.add_subplot(111, projection='3d')

        if plane is 'xy':
            for idx in range(cell.totnsegs):
                if idx in idx_ax:
                    if plot_axon:
                        axis.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], color=col_ax,
                                  alpha=alp, zorder=3)
                else:
                    if plot_dend:
                        axis.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], color=col_dend,
                                  alpha=alp, zorder=2)
            if plot_soma:
                e = Ellipse(xy=[cell.somapos[0], cell.somapos[2]],
                            width=somasize,
                            height=somasize,
                            color=col_soma,
                            zorder=10,
                            alpha=alp)
                axis.add_artist(e)

            axis.set_xlabel('x ($\mu$m)')
            axis.set_ylabel('y ($\mu$m)')

        elif plane is 'yz':
            for idx in range(cell.totnsegs):
                if idx in idx_ax:
                    if plot_axon:
                        axis.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], color=col_ax,
                                  alpha=alp, zorder=3)
                else:
                    if plot_dend:
                        axis.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], color=col_dend,
                                  alpha=alp, zorder=2)
            if plot_soma:
                e = Ellipse(xy=[cell.somapos[0], cell.somapos[2]],
                            width=somasize,
                            height=somasize,
                            color=col_soma,
                            zorder=10,
                            alpha=alp)
                axis.add_artist(e)

            axis.set_xlabel('y ($\mu$m)')
            axis.set_ylabel('z ($\mu$m)')

        elif plane is 'xz':
            for idx in range(cell.totnsegs):
                if idx in idx_ax:
                    if plot_axon:
                        axis.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], color=col_ax,
                                  alpha=alp, zorder=3)
                else:
                    if plot_dend:
                        axis.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], color=col_dend,
                                  alpha=alp, zorder=2)
            if plot_soma:
                e = Ellipse(xy=[cell.somapos[0], cell.somapos[2]],
                            width=somasize,
                            height=somasize,
                            color=col_soma,
                            zorder=10,
                            alpha=alp)
                axis.add_artist(e)

            axis.set_xlabel('x ($\mu$m)')
            axis.set_ylabel('z ($\mu$m)')

        elif plane is '3d':
            # ax = plt.gca()
            # ax_3d = plt.subplot(111, projection='3d')
            # ax.axis('off')
            ax_3d = axis
            for idx in range(cell.totnsegs):
                if idx == cell.somaidx:
                    for jj in range(len(cell.x3d[idx]) - 1):
                        init = np.array([cell.x3d[idx][jj], cell.y3d[idx][jj], cell.z3d[idx][jj]])
                        end = np.array([cell.x3d[idx][jj + 1], cell.y3d[idx][jj + 1], cell.z3d[idx][jj + 1]])
                        dir_seg = (end - init) / np.linalg.norm(end - init)
                        len_seg = np.linalg.norm(end - init)
                        n_points = 10.
                        neur_poly3d = get_polygons_for_cylinder(init,
                                                                 direction=dir_seg,
                                                                 length=len_seg,
                                                                 radius=cell.diam3d[idx][jj] / 2,
                                                                 n_points=n_points,
                                                                 facecolor=col_soma,
                                                                 edgecolor=col_soma,
                                                                 lw = 0.5,
                                                                 alpha=alp,
                                                                 flatten_along_zaxis=False)
                        for crt_poly3d in neur_poly3d:
                            ax_3d.add_collection3d(crt_poly3d)
                    # ax_3d.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                    #            [cell.zstart[idx], cell.zend[idx]], 'r', lw=15)
                elif idx in idx_ax:
                    init = np.array([cell.xstart[idx], cell.ystart[idx], cell.zstart[idx]])
                    end = np.array([cell.xend[idx], cell.yend[idx], cell.zend[idx]])
                    dir_seg = (end - init) / np.linalg.norm(end - init)
                    len_seg = np.linalg.norm(end - init)
                    n_points = 10.
                    neur_poly3d = get_polygons_for_cylinder(init,
                                                            direction=dir_seg,
                                                            length=len_seg,
                                                            radius=cell.diam[idx] / 2,
                                                            n_points=n_points,
                                                            facecolor=col_ax,
                                                            alpha=alp,
                                                            flatten_along_zaxis=False)
                    for crt_poly3d in neur_poly3d:
                        ax_3d.add_collection3d(crt_poly3d)
                    # ax_3d.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]],
                    #            [cell.zstart[idx], cell.zend[idx]], 'gray', lw=3)
                else:
                    # Add limit on x axis
                    if condition is not None:
                        if eval(condition):
                            init = np.array([cell.xstart[idx], cell.ystart[idx], cell.zstart[idx]])
                            end = np.array([cell.xend[idx], cell.yend[idx], cell.zend[idx]])
                            dir_seg = (end - init) / np.linalg.norm(end - init)
                            len_seg = np.linalg.norm(end - init)
                            n_points = 10.
                            neur_poly3d = get_polygons_for_cylinder(init,
                                                                    direction=dir_seg,
                                                                    length=len_seg,
                                                                    radius=cell.diam[idx] / 2,
                                                                    n_points=n_points,
                                                                    facecolor=col_dend,
                                                                    alpha=alp,
                                                                    flatten_along_zaxis=False)
                            for crt_poly3d in neur_poly3d:
                                ax_3d.add_collection3d(crt_poly3d)
                    else:
                        init = np.array([cell.xstart[idx], cell.ystart[idx], cell.zstart[idx]])
                        end = np.array([cell.xend[idx], cell.yend[idx], cell.zend[idx]])
                        dir_seg = (end - init) / np.linalg.norm(end - init)
                        len_seg = np.linalg.norm(end - init)
                        n_points = 10.
                        neur_poly3d = get_polygons_for_cylinder(init,
                                                                direction=dir_seg,
                                                                length=len_seg,
                                                                radius=cell.diam[idx] / 2,
                                                                n_points=n_points,
                                                                facecolor=col_dend,
                                                                alpha=alp,
                                                                flatten_along_zaxis=False)
                        for crt_poly3d in neur_poly3d:
                            ax_3d.add_collection3d(crt_poly3d)

            ax_3d.set_xlabel('x ($\mu$m)')
            ax_3d.set_ylabel('y ($\mu$m)')
            ax_3d.set_zlabel('z ($\mu$m)')
            axis = ax_3d

        if bounds is not None:
            if len(bounds) == 4:
                axis.axis('on')
                axis.set_xlim([bounds[0], bounds[1]])
                axis.set_ylim([bounds[2], bounds[3]])
                axis.axis('off')
        else:
            if plane is '3d':
                axis.set_xlim3d(np.min(cell.xmid), np.max(cell.xmid))
                axis.set_ylim3d(np.min(cell.ymid), np.max(cell.ymid))
                axis.set_zlim3d(np.min(cell.zmid), np.max(cell.zmid))
                axis.axis('equal')
            else:
                if xlim:
                    axis.set_xlim(xlim)
                if ylim:
                    axis.set_ylim(ylim)
                if not xlim and not ylim:
                    axis.axis('equal')

        #del cell
        # return axis



def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    templatename = None
    f = file("template.hoc", 'r')
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print 'template {} found!'.format(templatename)
            continue
    return templatename


def return_cell_shape(cell_name, cell_folder):

    cwd = os.getcwd()
    os.chdir(cell_folder)
    print cell_folder
    morphologyfile = os.listdir('morphology')[0]  # glob('morphology\\*')[0]
    cell = LFPy.Cell(morphology=join('morphology', morphologyfile), pt3d=True, delete_sections=False)
    os.chdir(cwd)

    return cell

def plot_probe(mea_pos, mea_pitch, shape='square', elec_dim=10, axis=None, xlim=None, ylim=None):
    from matplotlib.path import Path
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection

    if axis:
        ax = axis
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    n_elec = mea_pos.shape[0]

    y_pitch = mea_pitch[0]
    z_pitch = mea_pitch[1]


    elec_size = elec_dim / 2
    elec_size = (np.min([y_pitch,z_pitch]) - 0.3*np.min([y_pitch,z_pitch]))/2.
    elec_dim = (np.min([y_pitch,z_pitch]) - 0.3*np.min([y_pitch,z_pitch]))

    min_y = np.min(mea_pos[:,1])
    max_y = np.max(mea_pos[:,1])
    min_z = np.min(mea_pos[:,2])
    max_z = np.max(mea_pos[:,2])
    center_y = 0
    probe_height = 200
    probe_top = max_z + probe_height
    prob_bottom = min_z - probe_height
    prob_corner = min_z - 0.1*probe_height
    probe_left = min_y - 0.1*probe_height
    probe_right = max_y + 0.1*probe_height

    print min_y, max_y, min_z, max_z


    verts = [
        (min_y - 2*elec_dim, probe_top),  # left, bottom
        (min_y - 2*elec_dim, prob_corner),  # left, top
        (center_y, prob_bottom),  # right, top
        (max_y + 2*elec_dim, prob_corner),  # right, bottom
        (max_y + 2*elec_dim, probe_top),
        (min_y - 2 * elec_dim, max_z + 2 * elec_dim) # ignored
    ]

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]

    path = Path(verts, codes)

    patch = patches.PathPatch(path, facecolor='green', edgecolor='k', lw=0.5, alpha=0.3)
    ax.add_patch(patch)

    if shape == 'square':
        for e in range(n_elec):
            elec = patches.Rectangle((mea_pos[e, 1] - elec_size, mea_pos[e, 2] - elec_size), elec_dim,  elec_dim,
                                     alpha=0.7, facecolor='orange', edgecolor=[0.3, 0.3, 0.3], lw=0.5)

            ax.add_patch(elec)
    elif shape == 'circle':
        for e in range(n_elec):
            elec = patches.Circle((mea_pos[e, 1], mea_pos[e, 2]), elec_size,
                                     alpha=0.7, facecolor='orange', edgecolor=[0.3, 0.3, 0.3], lw=0.5)

            ax.add_patch(elec)

    ax.set_xlim(probe_left - 5*elec_dim, probe_right + 5*elec_dim)
    ax.set_ylim(prob_bottom - 5*elec_dim, probe_top + 5*elec_dim)
    # ax.axis('equal')

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def plot_mea_recording(spikes, mea_pos, mea_pitch, color='k', points=False, lw=1):
    # create mea
    number_electrode = mea_pos.shape[0]

    y_pos = np.unique(mea_pos[:, 1])
    z_pos = np.unique(mea_pos[:, 2])
    z_col = mea_pos[np.where(mea_pos[:, 1]==y_pos[len(y_pos)//2])[0], 2]
    y_row = mea_pos[np.where(mea_pos[:, 2]==z_pos[len(y_pos)//2])[0], 1]
    N_z = len(z_col)
    N_y = len(y_row)
    y_pitch = mea_pitch[0]
    z_pitch = mea_pitch[1]
    y_width = np.ptp(y_row)+y_pitch
    z_width = np.ptp(z_col)+z_pitch

    # plot spikes on grid
    fig = plt.figure()
    for el in range(number_electrode):
        # use add axes to adjust position and size
        yoffset = abs(np.min(mea_pos[:, 1])) + y_pitch/2.
        zoffset = abs(np.min(mea_pos[:, 2])) + z_pitch/2.
        w = 0.9*y_pitch/y_width
        h = 0.9*z_pitch/z_width
        l = 0.05+(mea_pos[el, 1] - y_pitch/2. + yoffset) / y_width * 0.9
        b = 0.05+(mea_pos[el, 2] - z_pitch/2. + zoffset) / z_width * 0.9

        rect = l, b, w, h

        ax = fig.add_axes(rect)
        if len(spikes.shape) == 3:  # multiple
            if points:
                ax.plot(np.transpose(spikes[:, el, :]), linestyle='-', marker='o', ms=2, lw=lw)
            else:
                ax.plot(np.transpose(spikes[:, el, :]), lw=lw)
        else:
            if points:
                ax.plot(spikes[el, :], color=color, linestyle='-', marker='o', ms=2, lw=lw)
            else:
                ax.plot(spikes[el, :], color=color, lw=lw)

        ax.set_ylim([np.min(spikes), np.max(spikes)])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    # return axis registered for neuron overlapping and mea bounds
    axneur = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    axneur.axis('off')
    bounds = [np.min(mea_pos[:, 1]), np.max(mea_pos[:, 1]), np.min(mea_pos[:, 2]), np.max(mea_pos[:, 2])]
    return fig, axneur, bounds


def plot_max_trace(spike, mea_dim=None, axis=None):
    #  check if number of spike is 1
    if len(spike.shape) == 3:
        print 'Plot one spike at a time!'
        return
    else:
        if axis:
            ax = axis
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # find max (shape is only 0 and 1!)
        max_idx = np.unravel_index(spike.argmax(), spike.shape)
        if (mea_dim[0] * mea_dim[1]) == spike.shape[0]:
            mea_values = spike[:, max_idx[1]].reshape((mea_dim[0],mea_dim[1]))
            im = ax.matshow(np.transpose(mea_values))
            # fig.colorbar(im)
        else:
            print 'MEA dimensions are wrong!'


def plot_min_trace(spike, mea_dim=None, axis=None, peak_image=None, cmap='jet', style='mat', origin='lower'):
    #  check if number of spike is 1
    if len(spike.shape) == 3:
        raise AttributeError('Plot one spike at a time!')
    else:
        if axis:
            ax = axis
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # find min (shape is only 0 and 1!)
        if peak_image is None:
            min_idx = np.unravel_index(spike.argmin(), spike.shape)
            if (mea_dim[0] * mea_dim[1]) == spike.shape[0]:
                mea_values = spike[:, min_idx[1]].reshape((mea_dim[0], mea_dim[1]))
                if style == 'mat':
                    im = ax.matshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                else:
                    im = ax.imshow(np.transpose(mea_values), cmap=cmap, origin=origin)
            else:
                print 'MEA dimensions are wrong!'
        else:
            im = ax.matshow(np.transpose(peak_image))

        ax.axis('off')
        return ax, im


def plot_weight(weight, mea_dim=None, axis=None, cmap='viridis', style='mat', origin='lower'):
    #  check if number of spike is 1
    if len(weight.shape) == 3:
        raise AttributeError('Plot one weight at a time!')
    else:
        if axis:
            ax = axis
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # find min (shape is only 0 and 1!)
        if (mea_dim[0] * mea_dim[1]) == weight.shape[0]:
            mea_values = weight.reshape((mea_dim[0], mea_dim[1]))
            if style == 'mat':
                im = ax.matshow(np.transpose(mea_values), cmap=cmap, origin=origin)
            else:
                im = ax.imshow(np.transpose(mea_values), cmap=cmap, origin=origin)
        else:
            print 'MEA dimensions are wrong!'

        ax.axis('off')
        return ax, im



def play_spike(spike, mea_dim, time=None, save=False, axis=None, file=None):
    #  check if number of spike is 1
    if len(spike.shape) == 3:
        print 'Plot one spike at a time!'
        return
    else:
        if time:
            inter = time
        else:
            inter = 20

    # if save:
    #     plt.switch_backend('agg')
    # else:
    #     plt.switch_backend('qt4agg')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    z_min = np.min(spike)
    z_max = np.max(spike)

    im0 = ax.matshow(np.zeros((mea_dim[0], mea_dim[1])), vmin=z_min, vmax=z_max)
    fig.colorbar(im0)
    ims = []

    if (mea_dim[0] * mea_dim[1]) == spike.shape[0]:
        for t in range(spike.shape[1]):
            ims.append([ax.matshow(np.transpose(spike[:, t].reshape((mea_dim[0], mea_dim[1]))),
                                   vmin=z_min, vmax=z_max)])

    im_ani = animation.ArtistAnimation(fig, ims, interval=inter, repeat_delay=2500, blit=True)

    if save:
        plt.switch_backend('agg')
        mywriter = animation.FFMpegWriter(fps=60)
        if file:
            im_ani.save(file, writer=mywriter)
        else:
            im_ani.save('spike.mp4', writer=mywriter)

    return im_ani

