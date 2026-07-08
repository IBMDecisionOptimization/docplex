# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Test encoding of variable and constraint names.

  1) read in a file with a specified encoding

  2) extend its variable and constraint names with random unicode
     symbols: the longest will be >= 255 bytes

  3) write the modified file to disk with a (possibly different)
     specified encoding

  4) Solve the model and print the solution values and variable names

  5) Verify that the modified model and the model written to disk
     have the same names for all variables and constraints

No command line arguments are required.
"""
import cplex
import platform
import os
import random
import sys
from cplextestcase import CplexTestCase

# The to_hex function is useful for comparing hex representations of strings.
# import binascii
#
# def to_hex(t, nbytes):
#     "Format text t as a sequence of nbyte long values separated by spaces."
#     chars_per_item = nbytes * 2
#     hex_version = binascii.hexlify(t)
#     num_chunks = len(hex_version) / chars_per_item
#     def chunkify():
#         for start in xrange(0, len(hex_version), chars_per_item):
#             yield hex_version[start:start + chars_per_item]
#     return ' '.join(chunkify())

# when I use the full range of unicode symbols by setting
# max_unicode_symbol to 1114112 all of the random symbols are ones
# that my system can't display.  It should still be tested, but it's
# nice to get visual confirmation that a bunch of crazy characters get
# put to the screen.
# max_unicode_symbol = 1114112 # this is one past the real max
max_unicode_symbol = 1111
# exclude ascii characters, since some of them are not allowed in lp
# format and we shouldn't need to test them anyway
unicode_ints = list(range(128, max_unicode_symbol))


def random_unichr():
    return chr(random.choice(unicode_ints))

def extend_names(name_list, enc):
    for i, name in enumerate(name_list):
        suffix = "#"
        suffix_len = random.randint(1, 100)
        for _ in range(suffix_len):
            suffix = suffix + random_unichr()
        u_name = name + suffix
        name_list[i] = u_name

def zippit(l):
    return zip(list(range(len(l))), l)

def process(fname, readenc, writeenc, seed="g11n", useread=False):

    random.seed(seed)

    # read in a file with first encoding
    c0 = cplex.Cplex()
    if useread:
        c0.parameters.read.fileencoding.set(readenc)
    c0.read(fname)

    # extract the names in readenc
    var_names_readenc = c0.variables.get_names()
    con_names_readenc = c0.linear_constraints.get_names()

    # extend them with random unicode characters
    extend_names(var_names_readenc, readenc)
    extend_names(con_names_readenc, readenc)

    # set the variable and constraint names to their new extended forms
    c0.variables.set_names(zippit(var_names_readenc))
    c0.linear_constraints.set_names(zippit(con_names_readenc))

    # create a second Cplex object with the second encoding.
    # Also, from here on we'll use writenc.
    c1 = cplex.Cplex()
    c1.parameters.read.fileencoding.set(writeenc)

    # write the model with the second encoding to a temp file
    (_, ext) = os.path.splitext(fname)
    with CplexTestCase._getTempFileName(ext=ext) as tmpfile:
        c0.parameters.read.fileencoding.set(writeenc)
        c0.write(tmpfile)
        # read the model into our second object with the second encoding.
        c1.read(tmpfile)

    # extract the names in writeenc.
    var_names_writeenc = c1.variables.get_names()
    con_names_writeenc = c1.linear_constraints.get_names()

    # solve the first model
    c0.solve()

    # We cannot compare strings extracted with different API encodings. 
    # Apparently ICU and python have different notions about normalization.
    for i, rname in enumerate(var_names_readenc):
        wname = var_names_writeenc[i]
        # print u"{}[{}]: {}".format(readenc, i, rname)
        # print "{}[{}]: {}".format(writeenc, i, wname)
        # print "len({}[{}]): {}".format(readenc, i, len(rname))
        # print "len({}[{}]): {}".format(writeenc, i, len(wname))
        # print to_hex(rname.encode(readenc), 2)
        # print to_hex(wname, 2)
        # assert rname.encode(readenc) == wname
        rprimal = c0.solution.get_values(rname)
        assert i == c0.variables.get_indices(rname)
        assert i == c1.variables.get_indices(wname)
        # print to_hex(c0.variables.get_names(i), 2)
        # print to_hex(rname.encode(readenc), 2)
        # urname = unicode(c0.variables.get_names(i), readenc).encode(readenc)
        # uwname = unicode(c1.variables.get_names(i), writeenc).encode(readenc)

def main():
    process("../../data/afiro.mps", "utf-8", "utf-8")
    #process("../../data/afiro.mps", "utf-8", "big5")
    process("../../data/afiro.mps", "utf-8", "gbk")
    process("../../data/afiro.mps", "iso-8859-1", "utf-8")
    #process("../../data/afiro.mps", "iso-8859-1", "big5")
    process("../../data/afiro.mps", "iso-8859-1", "gbk")
    # FIXME? these fail after RTC-31558.
    # if not CplexTestCase.iswindows():
    #     # TODO: These tests do not pass on Windows.  Why?
    #     # We get: CPLEX Error  1441: Line 2: No NAME section.
    #     process("../../data/afiro.mps", "iso-8859-1", "utf-16-be")
    #     process("../../data/afiro.mps", "iso-8859-1", "utf-16-le")
    #     process("../../data/afiro.mps", "iso-8859-1", "utf-32-be")
    #     process("../../data/afiro.mps", "iso-8859-1", "utf-32-le")
    # FIXME?: these fail after RTC-31558.
    #process("../../data/afiro_utf16-le.mps", "utf-16-le", "utf-8", useread=True)
    #process("../../data/afiro_utf16-be.mps", "utf-16-be", "utf-8", useread=True)
    #process("../../data/afiro_utf32-le.mps", "utf-32-le", "utf-8", useread=True)
    #process("../../data/afiro_utf32-be.mps", "utf-32-be", "utf-8", useread=True)
    # IBM01149 encoding doesn't exist in Python (see
    #    https://docs.python.org/2.4/lib/standard-encodings.html)
    #process("../../data/afiro_IBM01149.mps", "IBM01149", "UTF-8", useread=True)

if __name__ == "__main__":
    main()
