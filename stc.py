import os, sys, pathlib
import re

from pycparser import c_parser as cp, c_generator as cg, parse_file
from pycparser.c_parser import c_ast as ct

class EmptyStatement:
    def __init__(self):
        pass
    def __len__(self):
        return 0

class Statement:
    def __init__(self, stmt):
        self.stmt = stmt
    def __len__(self):
        return 1

class Compound:
    def __init__(self, stmts = []):
        self.stmts = stmts
    def __len__(self):
        return sum(map(len, self.stmts))
    def append(self, item):
        self.stmts.append(item)
    def appends(self, *items):
        for i in items:
            self.stmts.append(i)

class Loop:
    def __init__(self, *stmts):
        self.stmts = merge_structures(*stmts)
    def __len__(self):
        return len(self.stmts)

class Either:
    def __init__(self, *args):
        self.stmts = list(args)
    def __len__(self):
        if self.stmts is None: return 0

        w = 0
        for s in self.stmts:
             if s is not None:
                w += len(s)
        return w

class FindFnDefs(ct.NodeVisitor):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.defs = []

    def visit_FuncDef(self, node):
        self.defs.append(node)

class FindFnCalls(ct.NodeVisitor):
    def __init__(self, fnname):
        super(self.__class__, self).__init__()
        self.gen = cg.CGenerator().visit
        self.fnname = fnname
        self.check = False

    def visit_FuncCall(self, node):
        fnname = self.gen(node.name)

        if fnname == self.fnname:
            self.check = True
        
class ReplaceNone(ct.NodeVisitor):
    def __init__(self):
        super(self.__class__, self).__init__()
    def visit_Compound(self, node):
        if node.block_items is None: node.block_items = [ct.EmptyStatement()]
        else:
            for s in node.block_items:
                self.visit(s)
    def visit_If(self, node):
        if node.iftrue is None: node.iftrue = ct.EmptyStatement()
        if node.iffalse is None: node.iffalse = ct.EmptyStatement()
        self.visit(node.iftrue)
        self.visit(node.iffalse)
    def visit_For(self, node):
        if node.init is None: node.init = ct.EmptyStatement()
        if node.cond is None: node.cond = ct.EmptyStatement()
        if node.next is None: node.next = ct.EmptyStatement()
        self.visit(node.stmt)
    def visit_Return(self, node):
        if node.expr is None: node.expr = ct.EmptyStatement()

def merge_structures(*structs):
    stmts = []
    for s in filter(lambda x: x is not None, structs):
        if len(s) == 0:
            pass
        elif isinstance(s, Compound):
            stmts += s.stmts
        else:
            stmts.append(s)
    return Compound(stmts)
            
class FABuilder(ct.NodeVisitor):
    def __init__(self, fnname = None):
        super(self.__class__, self).__init__()
        self.gen = cg.CGenerator().visit
        self.fnname = fnname
    
    #def generic_visit(self, node):
    #    for k in getattr(node, '__slots__', [0, 0])[:-2]:
    #        setattr(node, k, self.visit(getattr(node, k)))
    #    return node
    
    def visit_FuncCall(self, node):
        fnname = self.gen(node.name)

        if fnname == "HPL_pdupdate": # do pdupdate
            fnname = "HPL_pdupdateNN"

        return Statement("fcall/" + fnname)

    def visit_ID(self, node): return EmptyStatement()
    def visit_Constant(self, node): return EmptyStatement()
    def visit_EmptyStatement(self, node): return EmptyStatement()
    def visit_Return(self, node): return self.visit(node.expr)
    def visit_Cast(self, node): return self.visit(node.expr)
    
    def visit_Assignment(self, node):
        return merge_structures(
            self.visit(node.lvalue),
            self.visit(node.rvalue),
            Statement("Assign/" + node.op))
    def visit_BinaryOp(self, node):
        return merge_structures(
            self.visit(node.left),
            self.visit(node.right),
            Statement("BinaryOp/" + node.op))
    def visit_UnaryOp(self, node):
        return merge_structures(
            self.visit(node.expr),
            Statement("UnaryOp/" + node.op))
    def visit_StructRef(self, node): # name.field
        return merge_structures(
            self.visit(node.name),
            self.visit(node.field))
    def visit_ArrayRef(self, node): # name[subscript]
        return merge_structures(
            self.visit(node.name),
            self.visit(node.subscript))
    def visit_Compound(self, node):
        return merge_structures(
            *(self.visit(s) for s in node.block_items)
        )

    def visit_ExprList(self, node):
        return merge_structures(
            *(self.visit(e) for e in node.exprs)
        )

    def visit_If(self, node):
        cond = self.visit(node.cond)
        iftrue = self.visit(node.iftrue)
        iffalse = self.visit(node.iffalse)

        return merge_structures(
            cond,
            Either(
                iftrue,
                iffalse
            )
        )

    def visit_TernaryOp(self, node):
        return self.visit_If(node)

    def visit_Switch(self, node):
        return merge_structures(
            self.visit(node.cond),
            Either(
                *(self.visit(s) for s in node.stmt.block_items)
            )
        )
    def visit_Case(self, node):
        return merge_structures(
            *(self.visit(s) for s in node.stmts)
        )
    def visit_Default(self, node):
        return merge_structures(
            *(self.visit(s) for s in node.stmts)
        )


    def visit_For(self, node):
        init = self.visit(node.init)
        cond = self.visit(node.cond)
        incr = self.visit(node.next)
        stmt = self.visit(node.stmt)

        return merge_structures(
            init,
            Loop(
                cond,
                merge_structures(stmt, incr)
            )
        )
    
    def visit_While(self, node):
        cond = self.visit(node.cond)
        stmt = self.visit(node.stmt)

        return Loop(
            cond,
            stmt
        )

    def visit_DoWhile(self, node):
        cond = self.visit(node.cond)
        stmt = self.visit(node.stmt)
        
        return Loop(
            cond,
            stmt
        )

def analyze_code(filename):
    deffind = FindFnDefs()
    repNone = ReplaceNone()
    
    fns = {}
    ast = parse_file(filename, use_cpp = False)
    deffind.visit(ast)
    
    for d in deffind.defs:
        fnname = d.decl.name
        repNone.visit(d)
        fns[fnname] = d.body
    return fns

def get_recursion_list(fns):
    l = []
    for k in fns:
        v = FindFnCalls(k)
        v.visit(fns[k])
        if v.check:
            l.append(k)
    return l

def weighted_stc(fns, stmt, meta, cache):
    if isinstance(stmt, EmptyStatement):
        return 0.0
    
    if isinstance(stmt, Statement):
        if "fcall/" in stmt.stmt:
            fnname = stmt.stmt[6:]
            if meta['fnptn'] is not None:
                if meta['fnptn'].match(fnname):
                    return 1.0
                elif fnname not in fns: # routines from other libraries
                    return 0.0
            else:
                if fnname not in fns:
                    return 1.0
            
            if fnname not in cache:
                cache[fnname] = weighted_stc(fns, fns[fnname], meta, cache)
            
            return cache[fnname]
        
        else: # other statements (+, -, etc.)
            return 1.0 if meta['fnptn'] is None else 0.0
    
    # code block
    if isinstance(stmt, Compound):
        s = 0.0
        for st in stmt.stmts:
            s += weighted_stc(fns, st, meta, cache)
        return s
    
    # conditionals (if, switch)
    if isinstance(stmt, Either):
        ws = list(
            map(lambda x:
                weighted_stc(fns, x, meta, cache), stmt.stmts))
        s = 0.0
        if sum(ws) == 0:
            return 0.0
        for i, w in enumerate(ws):
            s += w * w / sum(ws)
        return s
    
    # loops (for, while, do-while)
    if isinstance(stmt, Loop):
        loopfactor = meta['N']
        stc = weighted_stc(fns, stmt.stmts, meta, cache)
        return loopfactor * stc
    
def compute_stc(parsed,
                n, nb, procs,
                primitive_fn_name = None,
                main_fn_name = 'main'):
    
    if primitive_fn_name is not None:
        if type(primitive_fn_name) == str:
            primitive_fn_name = re.compile(primitive_fn_name)
    
    builder = FABuilder()
    
    recurse_list = get_recursion_list(parsed)
    
    p = {}
    for k, v in parsed.items():
        p[k] = builder.visit(v)
    
    cache = dict(
        (k, 0.0) for k in recurse_list
    )
    
    for k in recurse_list:
        cache[k] = weighted_stc(
            p, p[k], {
                'fnptn': primitive_fn_name,
                'N': n,
                'PQ': procs,
                'NB': nb
            }, cache)
        cache[k] *= n // procs
    
    return weighted_stc(p, p[main_fn_name], {
        'fnptn': primitive_fn_name,
        'N': n,
        'PQ': procs,
        'NB': nb
    }, cache)