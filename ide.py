import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
import re
import time
import threading
import math
import random
import sys
from functools import cmp_to_key
import importlib  # Added for pyrequire support

#######################
# Extra Math Functions
#######################
def clamp(x, mn, mx):
    return max(mn, min(x, mx))

def lerp(a, b, t):
    return a + (b - a) * t

def map_value(x, inmin, inmax, outmin, outmax):
    return (x - inmin) / (inmax - inmin) * (outmax - inmin) + outmin

def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

# Gradient vectors for noise
gradients = [(1,1,0), (-1,1,0), (1,-1,0), (-1,-1,0),
             (1,0,1), (-1,0,1), (1,0,-1), (-1,0,-1),
             (0,1,1), (0,-1,1), (0,1,-1), (0,-1,-1)]
p = [i for i in range(256)]
random.shuffle(p)
p = p * 2

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def grad(hash, x, y, z):
    g = gradients[hash % 12]
    return g[0] * x + g[1] * y + g[2] * z

def noise(x, y, z):
    X = int(math.floor(x)) & 255
    Y = int(math.floor(y)) & 255
    Z = int(math.floor(z)) & 255
    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)
    u = fade(x)
    v = fade(y)
    w = fade(z)
    A  = p[X] + Y
    AA = p[A] + Z
    AB = p[A + 1] + Z
    B  = p[X + 1] + Y
    BA = p[B] + Z
    BB = p[B + 1] + Z
    return lerp(w,
        lerp(v,
            lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
            lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))
        ),
        lerp(v,
            lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
            lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))
        )
    )

def cbrt(x):
    return x ** (1 / 3) if x >= 0 else -((-x) ** (1 / 3))

#######################
# Task Library Functions
#######################
from types import FunctionType

def task_spawn(func, *args):
    def wrapper():
        if hasattr(func, '__class__') and func.__class__.__name__ == "UserFunction":
            interpreter = _state.get('interpreter')
            if interpreter is None:
                raise Exception("No interpreter available for task.spawn for UserFunction")
            interpreter.call_function(func, *args)
        else:
            func(*args)
    t = threading.Thread(target=wrapper)
    t.start()
    return t

def task_defer(func, *args):
    def wrapper():
        if hasattr(func, '__class__') and func.__class__.__name__ == "UserFunction":
            interpreter = _state.get('interpreter')
            if interpreter is None:
                raise Exception("No interpreter available for task.defer for UserFunction")
            interpreter.call_function(func, *args)
        else:
            func(*args)
    t = threading.Timer(0, wrapper)
    t.start()
    return t

def task_delay(duration, func, *args):
    def wrapper():
        if hasattr(func, '__class__') and func.__class__.__name__ == "UserFunction":
            interpreter = _state.get('interpreter')
            if interpreter is None:
                raise Exception("No interpreter available for task.delay for UserFunction")
            interpreter.call_function(func, *args)
        else:
            func(*args)
    t = threading.Timer(duration, wrapper)
    t.start()
    return t

def task_desynchronize():
    pass

def task_synchronize():
    pass

def task_wait(duration):
    time.sleep(duration)
    return duration

def task_cancel(thread):
    if hasattr(thread, 'cancel'):
        thread.cancel()

#######################
# OS Library Functions
#######################
def os_clock():
    return time.perf_counter()

def os_date(fmt, t=None):
    if t is None:
        t = time.time()
    return time.strftime(fmt, time.localtime(t))

def os_difftime(t2, t1):
    return t2 - t1

def os_time(tbl=None):
    return time.time()

#######################
# Built-in ipairs function
#######################
def builtin_ipairs(tbl):
    if not isinstance(tbl, dict):
        raise Exception("ipairs expects a table")
    result = []
    i = 1
    while i in tbl:
        result.append((i, tbl[i]))
        i += 1
    return result

#######################
# Lexer
#######################
TT_NUMBER = 'NUMBER'
TT_STRING = 'STRING'
TT_IDENT = 'IDENT'
TT_KEYWORD = 'KEYWORD'
TT_OP = 'OP'
TT_EOF = 'EOF'

KEYWORDS = {
    'if', 'then', 'else', 'elseif', 'end', 'while', 'do', 'print',
    'local', 'true', 'false', 'for', 'in', 'and', 'or', 'not',
    'function', 'return', 'break', 'nil', 'repeat', 'until'
}

class Token:
    def __init__(self, type_, value, line=None):
        self.type = type_
        self.value = value
        self.line = line

    def __repr__(self):
        return f'Token({self.type}, {self.value}, line={self.line})'

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.current_char = self.text[self.pos] if self.text else None

    def error(self, msg):
        raise Exception(f"Lexer error on line {self.line}: {msg}")

    def advance(self):
        if self.current_char == "\n":
            self.line += 1
        self.pos += 1
        if self.pos >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos >= len(self.text):
            return None
        return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char in " \t\r":
            self.advance()

    def skip_comment(self):
        while self.current_char is not None and self.current_char != "\n":
            self.advance()

    def number(self):
        num_str = ''
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            num_str += self.current_char
            self.advance()
        return Token(TT_NUMBER, float(num_str) if '.' in num_str else int(num_str), self.line)

    def string(self):
        str_val = ''
        quote = self.current_char
        self.advance()
        while self.current_char is not None and self.current_char != quote:
            str_val += self.current_char
            self.advance()
        if self.current_char != quote:
            self.error("Unterminated string literal")
        self.advance()
        return Token(TT_STRING, str_val, self.line)

    def identifier(self):
        ident = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            ident += self.current_char
            self.advance()
        if ident in KEYWORDS:
            return Token(TT_KEYWORD, ident, self.line)
        return Token(TT_IDENT, ident, self.line)

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char == '-' and self.peek() == '-':
                self.advance(); self.advance()
                self.skip_comment()
                continue
            if self.current_char in " \t\r":
                self.skip_whitespace()
                continue
            if self.current_char == "\n":
                self.advance()
                continue
            if self.current_char.isdigit():
                return self.number()
            if self.current_char in ('"', "'"):
                return self.string()
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()
            if self.current_char == '~' and self.peek() == '=':
                self.advance(); self.advance()
                return Token(TT_OP, '~=', self.line)
            if self.current_char == '=' and self.peek() == '=':
                self.advance(); self.advance()
                return Token(TT_OP, '==', self.line)
            if self.current_char == '>' and self.peek() == '=':
                self.advance(); self.advance()
                return Token(TT_OP, '>=', self.line)
            if self.current_char == '<' and self.peek() == '=':
                self.advance(); self.advance()
                return Token(TT_OP, '<=', self.line)
            if self.current_char == '/' and self.peek() == '/':
                self.advance(); self.advance()
                return Token(TT_OP, '//', self.line)
            if self.current_char == '.' and self.peek() == '.':
                self.advance(); self.advance()
                return Token(TT_OP, '..', self.line)
            if self.current_char in "+-" and self.peek() == "=":
                op = self.current_char + "="
                self.advance(); self.advance()
                return Token(TT_OP, op, self.line)
            if self.current_char in "+-*/=(),<>{}.%^#[]:":
                ch = self.current_char
                self.advance()
                return Token(TT_OP, ch, self.line)
            self.error(f"Unknown character: {self.current_char}")
        return Token(TT_EOF, None, self.line)

    def tokenize(self):
        tokens = []
        token = self.get_next_token()
        while token.type != TT_EOF:
            tokens.append(token)
            token = self.get_next_token()
        tokens.append(token)
        return tokens

#######################
# AST Nodes
#######################
class AST:
    pass

class Number(AST):
    def __init__(self, value):
        self.value = value

class String(AST):
    def __init__(self, value):
        self.value = value

class Boolean(AST):
    def __init__(self, value):
        self.value = True if value == 'true' else False

class Nil(AST):
    def __init__(self):
        pass
    def __repr__(self):
        return "Nil"

class RepeatUntil(AST):
    def __init__(self, block, condition):
        self.block = block
        self.condition = condition

class TableIndex(AST):
    def __init__(self, table_expr, index_expr):
        self.table_expr = table_expr
        self.index_expr = index_expr

class UnaryOp(AST):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

# The Not node is retained for explicit "not" usage (handled only in unary_expr now)
class Not(AST):
    def __init__(self, expr):
        self.expr = expr

class Var(AST):
    def __init__(self, name, line=None):
        self.name = name
        self.line = line

class TableLiteral(AST):
    def __init__(self, entries):
        self.entries = entries

class TableAccess(AST):
    def __init__(self, table_expr, field):
        self.table_expr = table_expr
        self.field = field

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class Assignment(AST):
    def __init__(self, left, expr, local=False):
        self.left = left
        self.expr = expr
        self.local = local

class Print(AST):
    def __init__(self, expr):
        self.expr = expr

class If(AST):
    def __init__(self, cond, then_block, else_block=None):
        self.cond = cond
        self.then_block = then_block
        self.else_block = else_block

class While(AST):
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

class ForIn(AST):
    def __init__(self, vars, expr, block):
        self.vars = vars
        self.expr = expr
        self.block = block

class ForNumeric(AST):
    def __init__(self, var, start, end, step, block):
        self.var = var
        self.start = start
        self.end = end
        self.step = step
        self.block = block

class FunctionDef(AST):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class FunctionLiteral(AST):
    def __init__(self, params, body):
        self.params = params
        self.body = body

class Return(AST):
    def __init__(self, expr):
        self.expr = expr

class Call(AST):
    def __init__(self, func, args):
        self.func = func
        self.args = args

class Break(AST):
    pass

class Block(AST):
    def __init__(self, statements):
        self.statements = statements

#######################
# Parser
#######################
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def error(self, msg):
        line = self.current_token.line if self.current_token and self.current_token.line else "?"
        raise Exception(f"Parser error on line {line}: {msg}")

    def eat(self, token_type, value=None):
        if self.current_token.type == token_type and (value is None or self.current_token.value == value):
            self.advance()
        else:
            expected = f"{token_type}:{value}" if value else token_type
            self.error(f"Expected {expected} but got {self.current_token.type}:{self.current_token.value}")

    def advance(self):
        self.pos += 1
        if self.pos >= len(self.tokens):
            self.current_token = Token(TT_EOF, None)
        else:
            self.current_token = self.tokens[self.pos]

    def parse(self):
        stmts = self.statement_list()
        return Block(stmts)

    def statement_list(self):
        stmts = []
        while self.current_token.type != TT_EOF and self.current_token.value not in ('else', 'elseif', 'end', 'until'):
            stmt = self.statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def statement(self):
        if self.current_token.type == TT_KEYWORD:
            if self.current_token.value == 'local':
                if self.tokens[self.pos+1].type == TT_KEYWORD and self.tokens[self.pos+1].value == 'function':
                    return self.local_function_def()
                else:
                    return self.local_assignment()
            elif self.current_token.value == 'if':
                return self.if_statement()
            elif self.current_token.value == 'while':
                return self.while_statement()
            elif self.current_token.value == 'for':
                return self.for_statement()
            elif self.current_token.value == 'repeat':
                return self.repeat_until_statement()
            elif self.current_token.value == 'do':
                self.eat(TT_KEYWORD, 'do')
                block = self.statement_list()
                self.eat(TT_KEYWORD, 'end')
                return Block(block)
            elif self.current_token.value == 'print':
                return self.print_statement()
            elif self.current_token.value == 'function':
                return self.function_def()
            elif self.current_token.value == 'return':
                return self.return_statement()
            elif self.current_token.value == 'break':
                self.eat(TT_KEYWORD, 'break')
                return Break()
            else:
                return self.expr()
        if self.current_token.type == TT_IDENT or (self.current_token.type == TT_OP and self.current_token.value in ('{', '[')):
            start_pos = self.pos
            expr_node = self.expr()
            if self.current_token.type == TT_OP and self.current_token.value in ('=', '+=', '-='):
                self.pos = start_pos
                self.current_token = self.tokens[self.pos]
                return self.assignment()
            else:
                return expr_node
        self.error(f"Unexpected token: {self.current_token}")

    def local_function_def(self):
        self.eat(TT_KEYWORD, 'local')
        self.eat(TT_KEYWORD, 'function')
        if self.current_token.type == TT_IDENT:
            name = self.current_token.value
            self.eat(TT_IDENT)
            self.eat(TT_OP, '(')
            params = []
            if not (self.current_token.type == TT_OP and self.current_token.value == ')'):
                params.append(self.current_token.value)
                self.eat(TT_IDENT)
                while self.current_token.type == TT_OP and self.current_token.value == ',':
                    self.eat(TT_OP, ',')
                    params.append(self.current_token.value)
                    self.eat(TT_IDENT)
            self.eat(TT_OP, ')')
            body = self.statement_list()
            self.eat(TT_KEYWORD, 'end')
            return Assignment([Var(name)], [FunctionLiteral(params, Block(body))], local=True)
        else:
            self.error("Expected function name after 'local function'")

    def for_statement(self):
        self.eat(TT_KEYWORD, 'for')
        vars = []
        vars.append(Var(self.current_token.value, self.current_token.line))
        self.eat(TT_IDENT)
        while self.current_token.type == TT_OP and self.current_token.value == ',':
            self.eat(TT_OP, ',')
            vars.append(Var(self.current_token.value, self.current_token.line))
            self.eat(TT_IDENT)
        if self.current_token.type == TT_OP and self.current_token.value == '=':
            if len(vars) > 1:
                self.error("Numeric for loop cannot have multiple variables")
            return self.numeric_for_statement(vars[0].name)
        else:
            self.eat(TT_KEYWORD, 'in')
            expr = self.expr()
            self.eat(TT_KEYWORD, 'do')
            block = self.statement_list()
            self.eat(TT_KEYWORD, 'end')
            return ForIn(vars, expr, block)

    def numeric_for_statement(self, var_name):
        self.eat(TT_OP, '=')
        start_expr = self.expr()
        self.eat(TT_OP, ',')
        end_expr = self.expr()
        step_expr = None
        if self.current_token.type == TT_OP and self.current_token.value == ',':
            self.eat(TT_OP, ',')
            step_expr = self.expr()
        self.eat(TT_KEYWORD, 'do')
        block = self.statement_list()
        self.eat(TT_KEYWORD, 'end')
        return ForNumeric(Var(var_name), start_expr, end_expr, step_expr, block)

    def varlist(self):
        vars = [self.postfix_expr()]
        while self.current_token.type == TT_OP and self.current_token.value == ',':
            self.eat(TT_OP, ',')
            vars.append(self.postfix_expr())
        return vars

    def explist(self):
        exps = [self.expr()]
        while self.current_token.type == TT_OP and self.current_token.value == ',':
            self.eat(TT_OP, ',')
            exps.append(self.expr())
        return exps

    def local_assignment(self):
        self.eat(TT_KEYWORD, 'local')
        left = self.varlist()
        self.eat(TT_OP, '=')
        expr = self.explist()
        return Assignment(left, expr, local=True)

    def assignment(self):
        left = self.varlist()
        if self.current_token.type == TT_OP and self.current_token.value in ("=", "+=", "-="):
            op = self.current_token.value
            self.advance()
            if op == "=":
                expr = self.explist()
                return Assignment(left, expr, local=False)
            else:
                if len(left) != 1:
                    self.error("Compound assignment only supports single variable assignment")
                right_expr = self.expr()
                compound_op = op[0]
                new_expr = BinOp(left[0], Token(TT_OP, compound_op, left[0].line), right_expr)
                return Assignment(left, [new_expr], local=False)
        else:
            self.error("Expected assignment operator")

    def print_statement(self):
        self.eat(TT_KEYWORD, 'print')
        self.eat(TT_OP, '(')
        expr = self.expr()
        self.eat(TT_OP, ')')
        return Print(expr)

    def if_statement(self):
        self.eat(TT_KEYWORD, 'if')
        cond = self.expr()
        self.eat(TT_KEYWORD, 'then')
        then_block = self.statement_list()
        else_clause = None
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'elseif':
            else_clause = [self.parse_elseif()]
        elif self.current_token.type == TT_KEYWORD and self.current_token.value == 'else':
            self.eat(TT_KEYWORD, 'else')
            else_clause = self.statement_list()
        self.eat(TT_KEYWORD, 'end')
        return If(cond, then_block, else_clause)

    def parse_elseif(self):
        self.eat(TT_KEYWORD, 'elseif')
        cond = self.expr()
        self.eat(TT_KEYWORD, 'then')
        then_block = self.statement_list()
        else_clause = None
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'elseif':
            else_clause = [self.parse_elseif()]
        elif self.current_token.type == TT_KEYWORD and self.current_token.value == 'else':
            self.eat(TT_KEYWORD, 'else')
            else_clause = self.statement_list()
        return If(cond, then_block, else_clause)

    def while_statement(self):
        self.eat(TT_KEYWORD, 'while')
        cond = self.expr()
        self.eat(TT_KEYWORD, 'do')
        body = self.statement_list()
        self.eat(TT_KEYWORD, 'end')
        return While(cond, body)

    def repeat_until_statement(self):
        self.eat(TT_KEYWORD, 'repeat')
        block = self.statement_list()
        self.eat(TT_KEYWORD, 'until')
        cond = self.expr()
        return RepeatUntil(block, cond)

    def function_def(self):
        self.eat(TT_KEYWORD, 'function')
        if self.current_token.type == TT_IDENT:
            name = self.current_token.value
            self.eat(TT_IDENT)
            if self.current_token.type == TT_OP and self.current_token.value in ('.', ':'):
                op = self.current_token.value
                self.eat(TT_OP, op)
                field = self.current_token.value
                self.eat(TT_IDENT)
                self.eat(TT_OP, '(')
                params = []
                if op == ':':
                    params.append("self")
                if not (self.current_token.type == TT_OP and self.current_token.value == ')'):
                    params.append(self.current_token.value)
                    self.eat(TT_IDENT)
                    while self.current_token.type == TT_OP and self.current_token.value == ',':
                        self.eat(TT_OP, ',')
                        params.append(self.current_token.value)
                        self.eat(TT_IDENT)
                self.eat(TT_OP, ')')
                body = self.statement_list()
                self.eat(TT_KEYWORD, 'end')
                func_literal = FunctionLiteral(params, Block(body))
                return Assignment([TableAccess(Var(name), field)], [func_literal])
            else:
                self.eat(TT_OP, '(')
                params = []
                if not (self.current_token.type == TT_OP and self.current_token.value == ')'):
                    params.append(self.current_token.value)
                    self.eat(TT_IDENT)
                    while self.current_token.type == TT_OP and self.current_token.value == ',':
                        self.eat(TT_OP, ',')
                        params.append(self.current_token.value)
                        self.eat(TT_IDENT)
                self.eat(TT_OP, ')')
                body = self.statement_list()
                self.eat(TT_KEYWORD, 'end')
                return FunctionDef(name, params, Block(body))
        else:
            self.error("Expected function name after 'function'")

    def function_literal(self):
        self.eat(TT_KEYWORD, 'function')
        self.eat(TT_OP, '(')
        params = []
        if not (self.current_token.type == TT_OP and self.current_token.value == ')'):
            params.append(self.current_token.value)
            self.eat(TT_IDENT)
            while self.current_token.type == TT_OP and self.current_token.value == ',':
                self.eat(TT_OP, ',')
                params.append(self.current_token.value)
                self.eat(TT_IDENT)
        self.eat(TT_OP, ')')
        body = self.statement_list()
        self.eat(TT_KEYWORD, 'end')
        return FunctionLiteral(params, Block(body))

    def return_statement(self):
        self.eat(TT_KEYWORD, 'return')
        expr = None
        if not (self.current_token.type in (TT_KEYWORD, TT_EOF) or (self.current_token.type == TT_OP and self.current_token.value == 'end')):
            expr = self.expr()
        return Return(expr)

    def expr(self):
        return self.logical_expr()

    def logical_expr(self):
        node = self.equality_expr()
        while self.current_token.type == TT_KEYWORD and self.current_token.value in ('and', 'or'):
            op = self.current_token
            self.eat(TT_KEYWORD, op.value)
            right = self.equality_expr()
            node = BinOp(node, op, right)
        return node

    def equality_expr(self):
        node = self.relational_expr()
        while self.current_token.type == TT_OP and self.current_token.value in ('==', '~='):
            op = self.current_token
            self.eat(TT_OP, op.value)
            right = self.relational_expr()
            node = BinOp(node, op, right)
        return node

    def relational_expr(self):
        node = self.concat_expr()
        while self.current_token.type == TT_OP and self.current_token.value in ('<', '>', '<=', '>='):
            op = self.current_token
            self.eat(TT_OP, op.value)
            right = self.concat_expr()
            node = BinOp(node, op, right)
        return node

    def concat_expr(self):
        node = self.arith_expr()
        while self.current_token.type == TT_OP and self.current_token.value == '..':
            op = self.current_token
            self.eat(TT_OP, '..')
            right = self.arith_expr()
            node = BinOp(node, op, right)
        return node

    def arith_expr(self):
        node = self.term()
        while self.current_token.type == TT_OP and self.current_token.value in ('+', '-'):
            op = self.current_token
            self.eat(TT_OP, op.value)
            right = self.term()
            node = BinOp(node, op, right)
        return node

    def term(self):
        node = self.unary_expr()
        while self.current_token.type == TT_OP and self.current_token.value in ('*', '/', '%', '//'):
            op = self.current_token
            self.eat(TT_OP, op.value)
            right = self.unary_expr()
            node = BinOp(node, op, right)
        return node

    def unary_expr(self):
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'not':
            op = self.current_token
            self.eat(TT_KEYWORD, 'not')
            expr = self.unary_expr()
            return UnaryOp(op, expr)
        elif self.current_token.type == TT_OP and self.current_token.value == '#':
            op = self.current_token
            self.eat(TT_OP, '#')
            expr = self.unary_expr()
            return UnaryOp(op, expr)
        elif self.current_token.type == TT_OP and self.current_token.value == '-':
            op = self.current_token
            self.eat(TT_OP, '-')
            expr = self.power_expr()
            return UnaryOp(op, expr)
        else:
            return self.power_expr()

    def power_expr(self):
        node = self.postfix_expr()
        if self.current_token.type == TT_OP and self.current_token.value == '^':
            op = self.current_token
            self.eat(TT_OP, '^')
            right = self.unary_expr()
            node = BinOp(node, op, right)
        return node

    def postfix_expr(self):
        node = self.primary_expr()
        while True:
            if self.current_token.type == TT_OP and self.current_token.value == '(':
                self.eat(TT_OP, '(')
                args = []
                if not (self.current_token.type == TT_OP and self.current_token.value == ')'):
                    args.append(self.expr())
                    while self.current_token.type == TT_OP and self.current_token.value == ',':
                        self.eat(TT_OP, ',')
                        args.append(self.expr())
                self.eat(TT_OP, ')')
                node = Call(node, args)
            elif self.current_token.type == TT_OP and self.current_token.value == '.':
                self.eat(TT_OP, '.')
                field = self.current_token.value
                self.eat(TT_IDENT)
                node = TableAccess(node, field)
            elif self.current_token.type == TT_OP and self.current_token.value == ':':
                self.eat(TT_OP, ':')
                field = self.current_token.value
                self.eat(TT_IDENT)
                self.eat(TT_OP, '(')
                args = []
                if not (self.current_token.type == TT_OP and self.current_token.value == ')'):
                    args.append(self.expr())
                    while self.current_token.type == TT_OP and self.current_token.value == ',':
                        self.eat(TT_OP, ',')
                        args.append(self.expr())
                self.eat(TT_OP, ')')
                node = Call(TableAccess(node, field), [node] + args)
            elif self.current_token.type == TT_OP and self.current_token.value == '[':
                self.eat(TT_OP, '[')
                index_expr = self.expr()
                self.eat(TT_OP, ']')
                node = TableIndex(node, index_expr)
            else:
                break
        return node

    def primary_expr(self):
        tok = self.current_token
        if tok.type == TT_NUMBER:
            self.eat(TT_NUMBER)
            return Number(tok.value)
        elif tok.type == TT_STRING:
            self.eat(TT_STRING)
            return String(tok.value)
        elif tok.type == TT_KEYWORD and tok.value in ('true', 'false'):
            self.eat(TT_KEYWORD, tok.value)
            return Boolean(tok.value)
        elif tok.type == TT_KEYWORD and tok.value == 'nil':
            self.eat(TT_KEYWORD, 'nil')
            return Nil()
        # Removed duplicate handling of "not" here; it is now solely handled in unary_expr.
        elif tok.type == TT_KEYWORD and tok.value == 'function':
            return self.function_literal()
        elif tok.type == TT_IDENT:
            self.eat(TT_IDENT)
            return Var(tok.value, tok.line)
        elif tok.type == TT_OP and tok.value == '(':
            self.eat(TT_OP, '(')
            node = self.expr()
            self.eat(TT_OP, ')')
            return node
        elif tok.type == TT_OP and tok.value == '{':
            return self.table_literal()
        else:
            self.error("Unexpected token in primary expression: " + str(tok))

    def table_literal(self):
        self.eat(TT_OP, '{')
        entries = []
        while not (self.current_token.type == TT_OP and self.current_token.value == '}'):
            if self.current_token.type == TT_OP and self.current_token.value == '[':
                self.eat(TT_OP, '[')
                key_expr = self.expr()
                self.eat(TT_OP, ']')
                self.eat(TT_OP, '=')
                value = self.expr()
                entries.append((key_expr, value))
            elif (self.current_token.type == TT_IDENT and
                  self.tokens[self.pos + 1].type == TT_OP and
                  self.tokens[self.pos + 1].value == '='):
                key_val = self.current_token.value
                self.eat(TT_IDENT)
                self.eat(TT_OP, '=')
                value = self.expr()
                entries.append((String(key_val), value))
            else:
                value = self.expr()
                entries.append((None, value))
            if self.current_token.type == TT_OP and self.current_token.value == ',':
                self.eat(TT_OP, ',')
            else:
                break
        self.eat(TT_OP, '}')
        return TableLiteral(entries)

#######################
# Static Analysis for Undefined Variables
#######################
class UndefinedVarException(Exception):
    def __init__(self, line, message):
        self.line = line
        self.message = message
        super().__init__(message)

def static_check(node, env):
    if isinstance(node, Block):
        for stmt in node.statements:
            static_check(stmt, env)
    elif isinstance(node, Assignment):
        for exp in node.expr:
            static_check(exp, env)
        for left_item in node.left:
            if isinstance(left_item, Var):
                if not node.local and left_item.name not in env:
                    raise UndefinedVarException(left_item.line, f"Variable '{left_item.name}' is not defined; use local to define new variables")
                env[left_item.name] = True
            else:
                static_check(left_item, env)
    elif isinstance(node, Var):
        if node.name not in env:
            raise UndefinedVarException(node.line, f"Undefined variable: {node.name}")
    elif isinstance(node, BinOp):
        static_check(node.left, env)
        static_check(node.right, env)
    elif isinstance(node, Print):
        static_check(node.expr, env)
    elif isinstance(node, If):
        static_check(node.cond, env)
        static_check(Block(node.then_block), env.copy())
        if node.else_block:
            static_check(Block(node.else_block), env.copy())
    elif isinstance(node, While):
        static_check(node.cond, env)
        static_check(Block(node.body), env.copy())
    elif isinstance(node, RepeatUntil):
        static_check(Block(node.block), env.copy())
        static_check(node.condition, env)
    elif isinstance(node, ForIn):
        static_check(node.expr, env)
        new_env = env.copy()
        for var in node.vars:
            new_env[var.name] = True
        static_check(Block(node.block), new_env)
    elif isinstance(node, ForNumeric):
        static_check(node.start, env)
        static_check(node.end, env)
        if node.step is not None:
            static_check(node.step, env)
        new_env = env.copy()
        new_env[node.var.name] = True
        static_check(Block(node.block), new_env)
    elif isinstance(node, FunctionDef):
        env[node.name] = True
        new_env = env.copy()
        for param in node.params:
            new_env[param] = True
        static_check(node.body, new_env)
    elif isinstance(node, FunctionLiteral):
        new_env = env.copy()
        for param in node.params:
            new_env[param] = True
        static_check(node.body, new_env)
    elif isinstance(node, Return):
        if node.expr:
            static_check(node.expr, env)
    elif isinstance(node, Call):
        static_check(node.func, env)
        for arg in node.args:
            static_check(arg, env)
    elif isinstance(node, TableLiteral):
        for key, val in node.entries:
            if key is not None:
                static_check(key, env)
            static_check(val, env)
    elif isinstance(node, TableAccess):
        static_check(node.table_expr, env)
    elif isinstance(node, TableIndex):
        static_check(node.table_expr, env)
        static_check(node.index_expr, env)
    elif isinstance(node, Not):
        static_check(node.expr, env)
    elif isinstance(node, UnaryOp):
        static_check(node.expr, env)
    elif isinstance(node, Nil):
        pass

#######################
# Table Library Extensions
#######################
def table_clear(tbl):
    if not isinstance(tbl, dict):
        raise Exception("table.clear expects a table")
    keys = list(tbl.keys())
    for key in keys:
        tbl.pop(key, None)
    return None

def table_clone(t):
    if not isinstance(t, dict):
        raise Exception("table.clone expects a table")
    new_t = t.copy()
    new_t.pop("__frozen__", None)
    return new_t

def table_concat(t, sep="", i=1, j=None):
    if not isinstance(t, dict):
        raise Exception("table.concat expects a table")
    if j is None:
        j = table_maxn(t)
    result = ""
    for index in range(i, j + 1):
        if index in t:
            result += str(t[index])
            if index < j:
                result += sep
    return result

def table_create(count, value):
    try:
        count = int(count)
    except:
        raise Exception("table.create expects count to be a number")
    return {i: value for i in range(1, count + 1)}

def table_find(haystack, needle, init=1):
    if not isinstance(haystack, dict):
        raise Exception("table.find expects a table")
    max_index = table_maxn(haystack)
    for i in range(int(init), max_index + 1):
        if haystack.get(i) == needle:
            return i
    return None

def table_freeze(t):
    if not isinstance(t, dict):
        raise Exception("table.freeze expects a table")
    t["__frozen__"] = True
    return t

def table_isfrozen(t):
    if not isinstance(t, dict):
        raise Exception("table.isfrozen expects a table")
    return t.get("__frozen__", False)

def table_maxn(t):
    if not isinstance(t, dict):
        raise Exception("table.maxn expects a table")
    max_key = 0
    for key in t:
        if isinstance(key, int) or (isinstance(key, float) and key.is_integer()):
            k = int(key)
            if k > max_key:
                max_key = k
    return max_key

def table_move(src, a, b, t, dst=None):
    if not isinstance(src, dict):
        raise Exception("table.move expects src to be a table")
    if dst is None:
        dst = src
    for i in range(int(a), int(b) + 1):
        dst[int(t) + (i - int(a))] = src.get(i)
    return dst

def table_pack(*values):
    t = {i: value for i, value in enumerate(values, 1)}
    t["n"] = len(values)
    return t

def table_sort(t, comp=None):
    if not isinstance(t, dict):
        raise Exception("table.sort expects a table")
    n = table_maxn(t)
    arr = [t[i] for i in range(1, n + 1) if i in t]
    if comp is not None:
        def cmp_func(a, b):
            if comp(a, b):
                return -1
            elif comp(b, a):
                return 1
            else:
                return 0
        sorted_arr = sorted(arr, key=cmp_to_key(cmp_func))
    else:
        sorted_arr = sorted(arr)
    for i in range(1, len(sorted_arr) + 1):
        t[i] = sorted_arr[i - 1]
    return None

def table_insert(tbl, value, pos=None):
    if not isinstance(tbl, dict):
        raise Exception("table.insert expects a table")
    if pos is not None:
        try:
            pos = int(pos)
        except:
            raise Exception("table.insert position must be convertible to an integer")
    if pos is None:
        indices = [k for k in tbl.keys() if isinstance(k, int)]
        pos = max(indices) + 1 if indices else 1
        tbl[pos] = value
    else:
        indices = [k for k in tbl.keys() if isinstance(k, int)]
        max_index = max(indices) if indices else 0
        for i in range(max_index, pos - 1, -1):
            tbl[i + 1] = tbl.get(i)
        tbl[pos] = value
    tbl["n"] = max(pos, tbl.get("n", 0))

def table_remove(tbl, pos=None):
    if not isinstance(tbl, dict):
        raise Exception("table.remove expects a table")
    indices = [k for k in tbl.keys() if isinstance(k, int)]
    if not indices:
        return None
    if pos is None:
        pos = max(indices)
    if pos not in tbl:
        raise Exception("Index out of range in table.remove")
    value = tbl[pos]
    max_index = max(indices)
    for i in range(pos, max_index):
        tbl[i] = tbl.get(i + 1)
    if max_index in tbl:
        del tbl[max_index]
    return value

def table_unpack(t, i=1, j=None):
    if not isinstance(t, dict):
        raise Exception("table.unpack expects a table")
    if j is None:
        j = table_maxn(t)
    return tuple(t.get(k) for k in range(int(i), int(j) + 1))

def table_concat_original(tbl, sep=""):
    return table_concat(tbl, sep)

#######################
# String Library Extensions
#######################
def string_len(s):
    if not isinstance(s, str):
        raise Exception("string.len expects a string")
    return len(s)

def string_sub(s, i, j=None):
    if not isinstance(s, str):
        raise Exception("string.sub expects a string")
    length = len(s)
    if i < 0:
        i = length + i + 1
    if j is not None and j < 0:
        j = length + j + 1
    if j is None:
        return s[i - 1:]
    else:
        return s[i - 1:j]

def string_upper(s):
    if not isinstance(s, str):
        raise Exception("string.upper expects a string")
    return s.upper()

def string_lower(s):
    if not isinstance(s, str):
        raise Exception("string.lower expects a string")
    return s.lower()

def string_char(*args):
    try:
        return ''.join(chr(int(x)) for x in args)
    except Exception as e:
        raise Exception("string.char error: " + str(e))

def string_byte(s, i=1, j=None):
    if not isinstance(s, str):
        raise Exception("string.byte expects a string")
    if j is None:
        pos = i - 1
        if pos < 0 or pos >= len(s):
            return None
        return ord(s[pos])
    else:
        start = i - 1
        end = j
        return [ord(ch) for ch in s[start:end]]

def string_reverse(s):
    if not isinstance(s, str):
        raise Exception("string.reverse expects a string")
    return s[::-1]

def string_rep(s, n):
    if not isinstance(s, str):
        raise Exception("string.rep expects a string")
    try:
        n = int(n)
    except:
        raise Exception("string.rep expects an integer as second argument")
    return s * n

def string_find(s, pattern, init=1):
    if not isinstance(s, str):
        raise Exception("string.find expects a string")
    pos = s.find(pattern, init - 1)
    if pos == -1:
        return None
    else:
        return (pos + 1, pos + len(pattern))

def string_format(fmt, *args):
    try:
        return fmt % args
    except Exception as e:
        raise Exception("string.format error: " + str(e))

def string_gsub(s, pattern, repl, n=-1):
    if not isinstance(s, str):
        raise Exception("string.gsub expects a string")
    count = 0 if n < 0 else n
    result, num = re.subn(pattern, repl, s, count=count)
    return result, num

def builtin_select(first, *args):
    if first == "#":
        return len(args)
    else:
        try:
            n = int(first)
        except:
            raise Exception("select expects a number or '#' as the first argument")
        if n < 1 or n > len(args) + 1:
            return ()
        return args[n - 1:]

def builtin_type(val):
    if val is None:
        return "nil"
    elif isinstance(val, bool):
        return "boolean"
    elif isinstance(val, (int, float)):
        return "number"
    elif isinstance(val, str):
        return "string"
    elif isinstance(val, dict):
        return "table"
    elif callable(val):
        return "function"
    else:
        return "userdata"

def builtin_tostring(val):
    return str(val)

def builtin_tonumber(val):
    try:
        return float(val)
    except:
        return None

def builtin_ipairs(tbl):
    if not isinstance(tbl, dict):
        raise Exception("ipairs expects a table")
    result = []
    i = 1
    while i in tbl:
        result.append((i, tbl[i]))
        i += 1
    return result

#######################
# Exceptions and User Function
#######################
class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

class BreakException(Exception):
    pass

class UserFunction:
    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env.copy()

#######################
# (Game Library Extensions Removed)
#######################
# Note: All game and turtle-related functions and variables have been removed.

# _state will serve as a global dictionary for interpreter state
_state = {}

#######################
# Custom Python Module Loader (pyrequire)
#######################
def pyrequire(module_name):
    try:
        mod = importlib.import_module(module_name)
        return mod.__dict__
    except Exception as e:
        raise Exception("pyrequire failed: " + str(e))

#######################
# Interpreter and Built-in Functions
#######################
def math_random(*args):
    if len(args) == 0:
        return random.random()
    elif len(args) == 1:
        return random.randint(1, args[0])
    elif len(args) == 2:
        return random.randint(args[0], args[1])
    else:
        raise Exception("math.random expects 0, 1, or 2 arguments")

def math_randomseed(x):
    random.seed(x)

class Interpreter:
    def __init__(self, tree, output_callback=print):
        self.tree = tree
        self.env = {}
        self.output_callback = output_callback
        self.should_stop = False
        self.env['wait'] = self.builtin_wait
        self.env['require'] = self.builtin_require
        self.env['pyrequire'] = pyrequire  # Added pyrequire to environment
        self.env['pairs'] = self.builtin_pairs
        self.env['ipairs'] = builtin_ipairs
        self.env['select'] = builtin_select
        self.env['type'] = builtin_type
        self.env['tostring'] = builtin_tostring
        self.env['tonumber'] = builtin_tonumber
        self.env['math'] = {
            'abs': abs,
            'acos': math.acos,
            'asin': math.asin,
            'atan': math.atan,
            'atan2': math.atan2,
            'ceil': math.ceil,
            'clamp': clamp,
            'cos': math.cos,
            'cosh': math.cosh,
            'deg': math.degrees,
            'exp': math.exp,
            'floor': math.floor,
            'fmod': math.fmod,
            'frexp': math.frexp,
            'ldexp': math.ldexp,
            'lerp': lerp,
            'log': math.log,
            'log10': math.log10,
            'map': map_value,
            'max': max,
            'min': min,
            'modf': math.modf,
            'noise': noise,
            'pow': math.pow,
            'rad': math.radians,
            'round': round,
            'sign': sign,
            'sin': math.sin,
            'tanh': math.tanh,
            'sqrt': math.sqrt,
            'cbrt': cbrt,
            'huge': sys.float_info.max,
            'pi': math.pi,
            'random': math_random,
            'randomseed': math_randomseed
        }
        self.env['table'] = {
            'clear': table_clear,
            'clone': table_clone,
            'concat': table_concat,
            'create': table_create,
            'find': table_find,
            'freeze': table_freeze,
            'isfrozen': table_isfrozen,
            'maxn': table_maxn,
            'move': table_move,
            'insert': table_insert,
            'remove': table_remove,
            'pack': table_pack,
            'sort': table_sort,
            'unpack': table_unpack
        }
        self.env['string'] = {
            'len': string_len,
            'sub': string_sub,
            'upper': string_upper,
            'lower': string_lower,
            'char': string_char,
            'byte': string_byte,
            'reverse': string_reverse,
            'rep': string_rep,
            'find': string_find,
            'format': string_format,
            'gsub': string_gsub
        }
        self.env['task'] = {
            'spawn': task_spawn,
            'defer': task_defer,
            'delay': task_delay,
            'desynchronize': task_desynchronize,
            'synchronize': task_synchronize,
            'wait': task_wait,
            'cancel': task_cancel
        }
        self.env['os'] = {
            'clock': os_clock,
            'date': os_date,
            'difftime': os_difftime,
            'time': os_time
        }
        _state['interpreter'] = self
        self.start_time = time.time()

    def check_timeout(self):
        pass

    def call_function(self, func, *args):
        if len(args) != len(func.params):
            self.error(f"Function expected {len(func.params)} args, got {len(args)}")
        new_env = func.env.copy()
        for param, arg in zip(func.params, args):
            new_env[param] = arg
        old_env = self.env
        self.env = new_env
        try:
            self.visit(func.body)
        except ReturnException as ret:
            result = ret.value
        else:
            result = None
        self.env = old_env
        return result

    def builtin_wait(self, t):
        try:
            t = float(t)
        except Exception:
            self.error("wait expects a number")
        start = time.time()
        while time.time() - start < t:
            if self.should_stop:
                self.error("Execution stopped.")
            time.sleep(0.01)
            self.check_timeout()

    def builtin_require(self, filename):
        if not isinstance(filename, str):
            self.error("require expects a string filename")
        try:
            with open(filename, "r") as f:
                code = f.read()
        except Exception as e:
            self.error("Cannot load module: " + str(e))
        doc_dict = {}
        for line in code.splitlines():
            m = re.search(r'function\s+(\w+)\s*\(.*\)\s*--\s*(.+)$', line)
            if m:
                func_name = m.group(1)
                doc = m.group(2).strip()
                doc_dict[func_name] = doc
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        tree = parser.parse()
        module_interpreter = Interpreter(tree, output_callback=lambda x: None)
        module_interpreter.env['wait'] = self.builtin_wait
        module_interpreter.env['require'] = self.builtin_require
        try:
            module_interpreter.run()
        except ReturnException as ret:
            module_value = ret.value
        else:
            module_value = module_interpreter.env
        if isinstance(module_value, dict):
            module_value["__doc__"] = doc_dict
        return module_value

    def builtin_pairs(self, tbl):
        if isinstance(tbl, dict):
            return list(tbl.items())
        self.error("pairs expects a table (dict)")

    def error(self, msg):
        raise Exception("Runtime error: " + msg)

    def run(self):
        self.start_time = time.time()
        self.visit(self.tree)

    def visit(self, node):
        if self.should_stop:
            self.error("Execution stopped.")
        self.check_timeout()
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        self.error(f"No visit_{type(node).__name__} method")

    def visit_Block(self, node):
        for stmt in node.statements:
            if self.should_stop:
                self.error("Execution stopped.")
            try:
                self.visit(stmt)
            except BreakException:
                raise BreakException()

    def visit_Number(self, node):
        return node.value

    def visit_String(self, node):
        return node.value

    def visit_Boolean(self, node):
        return node.value

    def visit_Nil(self, node):
        return None

    def visit_Var(self, node):
        if node.name in self.env:
            return self.env[node.name]
        self.error(f"Variable '{node.name}' is not defined")

    def visit_TableLiteral(self, node):
        table = {}
        array_index = 1
        for key, value in node.entries:
            val = self.visit(value)
            if key is None:
                table[array_index] = val
                array_index += 1
            else:
                table[self.visit(key)] = val
        return table

    def visit_TableAccess(self, node):
        table = self.visit(node.table_expr)
        field = node.field
        if isinstance(table, dict):
            if field in table:
                return table[field]
            self.error(f"Field '{field}' not found in table")
        elif isinstance(table, str):
            string_lib = self.env.get('string', {})
            if field in string_lib:
                return lambda *args: string_lib[field](table, *args)
            self.error(f"Method '{field}' not found for string")
        else:
            self.error(f"Cannot access field '{field}' of non-table and non-string value")

    def visit_TableIndex(self, node):
        table = self.visit(node.table_expr)
        index = self.visit(node.index_expr)
        if isinstance(table, dict):
            return table.get(index, None)
        else:
            self.error("Attempt to index a non-table value")

    def visit_Not(self, node):
        return not self.visit(node.expr)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.expr)
        if node.op.value == '-':
            return -operand
        elif node.op.value == '#':
            if isinstance(operand, str):
                return len(operand)
            elif isinstance(operand, dict):
                count = 0
                while (count + 1) in operand:
                    count += 1
                return count
            else:
                self.error("Operator '#' not defined for this type")
        elif node.op.value == 'not':
            return not operand
        else:
            self.error(f"Unknown unary operator {node.op.value}")

    def visit_BinOp(self, node):
        op = node.op.value
        if op == 'and':
            left = self.visit(node.left)
            if not left:
                return left
            return self.visit(node.right)
        elif op == 'or':
            left = self.visit(node.left)
            if left:
                return left
            return self.visit(node.right)
        else:
            left = self.visit(node.left)
            right = self.visit(node.right)
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return left / right
            elif op == '%':
                return left % right
            elif op == '//':
                return left // right
            elif op == '^':
                return left ** right
            elif op == '<':
                return left < right
            elif op == '>':
                return left > right
            elif op == '<=':
                return left <= right
            elif op == '>=':
                return left >= right
            elif op == '==':
                return left == right
            elif op == '~=':
                return left != right
            elif op == '..':
                return str(left) + str(right)
            else:
                self.error(f"Unknown binary operator {op}")

    def visit_Assignment(self, node):
        values = [self.visit(exp) for exp in node.expr]
        if len(values) < len(node.left):
            values.extend([None] * (len(node.left) - len(values)))
        for var_node, value in zip(node.left, values):
            if isinstance(var_node, Var):
                if not node.local and var_node.name not in self.env:
                    self.error("Variable '" + var_node.name + "' is not defined; use local to define new variables")
                self.env[var_node.name] = value
            elif isinstance(var_node, TableAccess):
                target = self.visit(var_node.table_expr)
                target[var_node.field] = value
            elif isinstance(var_node, TableIndex):
                target = self.visit(var_node.table_expr)
                index = self.visit(var_node.index_expr)
                target[index] = value
            else:
                self.error("Invalid assignment target")
        return values[-1] if values else None

    def visit_Print(self, node):
        val = self.visit(node.expr)
        self.output_callback(str(val))
        return val

    def visit_If(self, node):
        cond = self.visit(node.cond)
        if cond:
            for stmt in node.then_block:
                self.visit(stmt)
        elif node.else_block:
            for stmt in node.else_block:
                self.visit(stmt)

    def visit_While(self, node):
        while self.visit(node.cond):
            try:
                for stmt in node.body:
                    self.visit(stmt)
            except BreakException:
                break

    def visit_RepeatUntil(self, node):
        while True:
            for stmt in node.block:
                self.visit(stmt)
            if self.visit(node.condition):
                break

    def visit_ForIn(self, node):
        iterable = self.visit(node.expr)
        if not hasattr(iterable, '__iter__'):
            self.error("Value is not iterable")
        for item in iterable:
            if len(node.vars) == 1:
                self.env[node.vars[0].name] = item
            else:
                if not isinstance(item, (list, tuple)):
                    self.error("Iterator did not return multiple values")
                if len(item) != len(node.vars):
                    self.error("Iterator did not return expected number of values")
                for var, val in zip(node.vars, item):
                    self.env[var.name] = val
            try:
                for stmt in node.block:
                    self.visit(stmt)
            except BreakException:
                break

    def visit_ForNumeric(self, node):
        start = self.visit(node.start)
        end_val = self.visit(node.end)
        step = self.visit(node.step) if node.step is not None else 1
        i = start
        if step > 0:
            while i <= end_val:
                self.env[node.var.name] = i
                try:
                    for stmt in node.block:
                        self.visit(stmt)
                except BreakException:
                    break
                i += step
        else:
            while i >= end_val:
                self.env[node.var.name] = i
                try:
                    for stmt in node.block:
                        self.visit(stmt)
                except BreakException:
                    break
                i += step

    def visit_FunctionDef(self, node):
        func = UserFunction(node.params, node.body, self.env)
        self.env[node.name] = func
        return None

    def visit_FunctionLiteral(self, node):
        return UserFunction(node.params, node.body, self.env.copy())

    def visit_Return(self, node):
        value = self.visit(node.expr) if node.expr else None
        raise ReturnException(value)

    def visit_Call(self, node):
        func_val = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        if callable(func_val):
            return func_val(*args)
        elif isinstance(func_val, UserFunction):
            if len(args) != len(func_val.params):
                self.error(f"Function expected {len(func_val.params)} args, got {len(args)}")
            new_env = func_val.env.copy()
            for param, arg in zip(func_val.params, args):
                new_env[param] = arg
            old_env = self.env
            self.env = new_env
            try:
                self.visit(func_val.body)
            except ReturnException as ret:
                result = ret.value
            else:
                result = None
            self.env = old_env
            return result
        else:
            self.error("Attempt to call non-function value")

    def visit_Break(self, node):
        raise BreakException()

#######################
# Tkinter IDE with Enhanced Features
#######################
class LuaIDE(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Enhanced Lua Interpreter IDE")
        self.geometry("900x700")
        self.configure(bg="#21252B")

        # Removed "game" from completion words
        self.completion_words = list(KEYWORDS) + ['wait', 'require', 'pyrequire', 'pairs', 'ipairs', 'select', 'type', 'tostring',
                                                  'tonumber', 'math', 'table', 'string', 'task', 'os']
        self.completion_box = None
        self.current_interpreter = None
        self.run_thread = None

        menubar = tk.Menu(self, bg="#21252B", fg="white")
        filemenu = tk.Menu(menubar, tearoff=0, bg="#21252B", fg="white")
        filemenu.add_command(label="Clear", command=self.clear_editor)
        filemenu.add_command(label="Save", command=self.save_file)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

        self.editor = ScrolledText(self, height=20, font=("Consolas", 14),
                                   bg="#282C34", fg="#ABB2BF", insertbackground="white",
                                   undo=True, wrap=tk.NONE)
        self.editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        sample = (
            "-- UPDATE 4.0.0 --\n"
            "-- Ask gpt for help if needed -- \n"
            "-- This is a comment using '--' as in real Lua\n"
            "local e = { name = \"hero\", attack = function(self) -- Performs an attack\n"
            "    print(self.name .. \" attacks!\")\n"
            "    return 10\n"
            "end }\n\n"
            "function double(x) -- Doubles the input value\n"
            "    return x * 2\n"
            "end\n\n"
            "-- Table member function definition using colon syntax\n"
            "function e:heal(amount)\n"
            "    print(self.name .. \" heals for \" .. amount)\n"
            "end\n\n"
            "if e.name == \"hero\" then\n"
            "    print(\"Welcome, \" .. e.name)\n"
            "elseif e.name == \"villain\" then\n"
            "    print(\"Beware of \" .. e.name)\n"
            "else\n"
            "    print(\"Unknown character\")\n"
            "end\n\n"
            "local damage = e.attack(e)\n"
            "print(\"Damage dealt: \" .. double(damage))\n\n"
            "-- Calling a method using colon syntax\n"
            "e:heal(15)\n\n"
            "-- Multiple assignment example:\n"
            "local x, y = 1, 2\n"
            "print(\"x: \" .. x .. \", y: \" .. y)\n\n"
            "-- Compound assignment example:\n"
            "x += 1\n"
            "print(\"x after += 1: \" .. x)\n\n"
            "-- select() example:\n"
            "print(\"Number of extra arguments: \" .. select(\"#\", 10, 20, 30))\n"
            "local extras = {select(2, 10, 20, 30)}\n"
            "print(\"Extras: \" .. table.concat(extras, \",\"))\n\n"
            "-- Numeric for loop example:\n"
            "for i = 1, 5, 1 do\n"
            "    print(i)\n"
            "end\n\n"
            "-- For-in loop example using pairs:\n"
            "for key, value in pairs({a=1, b=2, c=3}) do\n"
            "    print(key .. \" : \"  .. value)\n"
            "end\n\n"
            "-- For-in loop example using ipairs:\n"
            "for index, value in ipairs({10, 20, 30}) do\n"
            "    print(index .. \" => \" .. value)\n"
            "end\n\n"
            "-- Repeat-until loop example:\n"
            "local count = 0\n"
            "repeat\n"
            "    count = count + 1\n"
            "    print(\"Count: \" .. count)\n"
            "until count >= 3\n\n"
            "-- Math library extended examples:\n"
            "print(\"clamp(15, 0, 10): \" .. math.clamp(15, 0, 10))\n"
            "print(\"lerp(0, 100, 0.5): \" .. math.lerp(0, 100, 0.5))\n"
            "print(\"map(5, 0, 10, 0, 100): \" .. math.map(5, 0, 10, 0, 100))\n"
            "print(\"sign(-10): \" .. math.sign(-10))\n"
            "print(\"sin(1): \" .. math.sin(1))\n"
            "print(\"tanh(1): \" .. math.tanh(1))\n"
            "print(\"sqrt(16): \" .. math.sqrt(16))\n"
            "print(\"cbrt(27): \" .. math.cbrt(27))\n\n"
            "-- Task library examples:\n"
            "task.spawn(function() print(\"Hello from spawn!\") end)\n"
            "task.delay(1, function() print(\"Hello after 1 second delay\") end)\n\n"
            "-- OS library examples:\n"
            "print(\"os.clock: \" .. os.clock())\n"
            "print(\"os.date: \" .. os.date(\"%Y-%m-%d %H:%M:%S\"))\n"
            "print(\"os.difftime: \" .. os.difftime(os.time(), 0))\n"
            "print(\"os.time: \" .. os.time())\n\n"
            "-- Example of pyrequire usage:\n"
            "-- Assuming you have a python module named 'my_module' accessible in PYTHONPATH,\n"
            "-- you can load it as follows and call its functions from Lua:\n"
            "-- local mod = pyrequire('my_module')\n"
            "-- print(\"Module functions: \" .. tostring(mod))\n"
        )
        self.editor.insert(tk.END, sample)

        self.editor.bind("<KeyRelease>", self.on_key_release)
        self.editor.bind("<Return>", self.handle_return)

        button_frame = tk.Frame(self, bg="#21252B")
        button_frame.pack(pady=5)
        self.run_button = tk.Button(button_frame, text="Run", command=self.run_code,
                                    font=("Consolas", 12), bg="#61AFEF", fg="black")
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_code,
                                     font=("Consolas", 12), bg="#E06C75", fg="black")
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.output = ScrolledText(self, height=10, font=("Consolas", 14),
                                   bg="#21252B", fg="white", state=tk.NORMAL)
        self.output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.setup_tags()

    def setup_tags(self):
        self.editor.tag_configure("keyword", foreground="#C678DD")  # Purple
        self.editor.tag_configure("string", foreground="#98C379")  # Green
        self.editor.tag_configure("number", foreground="#D19A66")  # Orange
        self.editor.tag_configure("boolean", foreground="#E5C07B")  # Yellow
        self.editor.tag_configure("comment", foreground="#7F848E", font=("Consolas", 14, "italic"))  # Grey
        self.editor.tag_configure("operator", foreground="#56B6C2")  # Cyan
        self.editor.tag_configure("function", foreground="#61AFEF")  # Blue
        self.editor.tag_configure("variable", foreground="#E06C75")  # Red
        self.editor.tag_configure("error", underline=True, foreground="#FF5555")  # Bright Red
        self.editor.tag_configure("undefined", underline=True, foreground="#61AFEF")  # Blue
        self.editor.tag_configure("local_keyword", foreground="#D19A66")  # Orange (for "local")

    def clear_editor(self):
        self.editor.delete(1.0, tk.END)

    def save_file(self):
        filename = filedialog.asksaveasfilename(defaultextension=".lua",
                                                filetypes=[("Lua files", "*.lua"), ("All files", "*.*")])
        if filename:
            code = self.editor.get("1.0", tk.END)
            with open(filename, "w") as f:
                f.write(code)

    def append_output(self, text):
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)

    def run_code(self):
        code = self.editor.get(1.0, tk.END)
        self.output.delete(1.0, tk.END)

        def run_interpreter():
            try:
                lexer = Lexer(code)
                tokens = lexer.tokenize()
                parser = Parser(tokens)
                tree = parser.parse()
                # Removed: "game" from static check list
                static_check(tree, {"wait": True, "require": True, "pyrequire": True, "pairs": True, "ipairs": True, "select": True,
                                    "type": True, "tostring": True, "tonumber": True, "math": True, "table": True,
                                    "string": True, "task": True, "os": True})
                self.current_interpreter = Interpreter(tree, output_callback=self.append_output)
                self.current_interpreter.should_stop = False
                self.current_interpreter.env['require'] = self.current_interpreter.builtin_require
                self.current_interpreter.run()
            except Exception as e:
                self.append_output("Error: " + str(e))

        self.run_thread = threading.Thread(target=run_interpreter)
        self.run_thread.start()

    def stop_code(self):
        # Improved stop: signal stop, wait briefly for thread termination,
        # clear current interpreter and reset interpreter state to refresh Python environment.
        if self.current_interpreter:
            self.current_interpreter.should_stop = True
            if self.run_thread:
                self.run_thread.join(timeout=1)
            self.current_interpreter = None
            _state.clear()
            self.append_output("Interpreter stopped and environment refreshed.")

    def on_key_release(self, event):
        self.highlight_syntax()
        self.show_autocomplete()
        self.check_syntax()

    def highlight_syntax(self):
        content = self.editor.get("1.0", tk.END)
        self.editor.tag_remove("keyword", "1.0", tk.END)
        self.editor.tag_remove("string", "1.0", tk.END)
        self.editor.tag_remove("number", "1.0", tk.END)
        self.editor.tag_remove("boolean", "1.0", tk.END)
        self.editor.tag_remove("comment", "1.0", tk.END)
        self.editor.tag_remove("operator", "1.0", tk.END)
        self.editor.tag_remove("function", "1.0", tk.END)
        self.editor.tag_remove("variable", "1.0", tk.END)
        self.editor.tag_remove("local_keyword", "1.0", tk.END)

        # Highlight keywords (excluding "local")
        for kw in KEYWORDS - {"local"}:
            start = "1.0"
            while True:
                pos = self.editor.search(r'\b' + kw + r'\b', start, stopindex=tk.END, regexp=True)
                if not pos:
                    break
                end = f"{pos}+{len(kw)}c"
                self.editor.tag_add("keyword", pos, end)
                start = end

        # Highlight "local" separately in orange
        start = "1.0"
        while True:
            pos = self.editor.search(r'\blocal\b', start, stopindex=tk.END, regexp=True)
            if not pos:
                break
            end = f"{pos}+5c"
            self.editor.tag_add("local_keyword", pos, end)
            start = end

        # Highlight booleans
        for match in re.finditer(r'\b(true|false|nil)\b', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("boolean", start_index, end_index)

        # Highlight strings
        for match in re.finditer(r'(\".*?\"|\'.*?\')', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("string", start_index, end_index)

        # Highlight numbers
        for match in re.finditer(r'\b\d+(\.\d+)?\b', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("number", start_index, end_index)

        # Highlight comments
        for match in re.finditer(r'--.*', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("comment", start_index, end_index)

        # Highlight operators
        for match in re.finditer(r'(\+|\-|\*|\/|\=|\~\=|\=\=|\<|\>|\<=|\>=|\.\.)', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("operator", start_index, end_index)

        # Highlight function names
        for match in re.finditer(r'\bfunction\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', content):
            start_index = self.index_from_pos(match.start(1))
            end_index = self.index_from_pos(match.end(1))
            self.editor.tag_add("function", start_index, end_index)

        # Highlight variables (excluding "local")
        for match in re.finditer(r'\blocal\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', content):
            start_index = self.index_from_pos(match.start(1))
            end_index = self.index_from_pos(match.end(1))
            self.editor.tag_add("variable", start_index, end_index)

    def index_from_pos(self, pos):
        return "1.0+" + str(pos) + "c"

    def show_autocomplete(self):
        # Minimal autocomplete (implementation placeholder)
        pass

    def check_syntax(self):
        self.editor.tag_remove("error", "1.0", tk.END)
        self.editor.tag_remove("undefined", "1.0", tk.END)
        code = self.editor.get("1.0", tk.END)
        try:
            lexer = Lexer(code)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            static_check(ast,
                         {"wait": True, "require": True, "pyrequire": True, "pairs": True, "ipairs": True, "select": True, "type": True,
                          "tostring": True, "tonumber": True, "math": True, "table": True, "string": True,
                          "task": True, "os": True})
        except Exception as e:
            err_msg = str(e)
            m = re.search(r'line (\d+)', err_msg)
            if m:
                error_line = m.group(1)
                self.editor.tag_add("error", f"{error_line}.0", f"{error_line}.end")

    def handle_return(self, event):
        self.on_key_release(event)
        return None

if __name__ == "__main__":
    ide = LuaIDE()
    ide.mainloop()
