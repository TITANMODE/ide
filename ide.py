import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
import re
import time
import threading
import math
import random

#######################
# Lexer
#######################
TT_NUMBER = 'NUMBER'
TT_STRING = 'STRING'
TT_IDENT = 'IDENT'
TT_KEYWORD = 'KEYWORD'
TT_OP = 'OP'
TT_EOF = 'EOF'

# Keywords (added "nil", "repeat", "until", and "elseif")
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
        # Skip until end of line (only supports '--' comments as in Lua)
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
        self.advance()  # skip opening quote
        while self.current_char is not None and self.current_char != quote:
            str_val += self.current_char
            self.advance()
        if self.current_char != quote:
            self.error("Unterminated string literal")
        self.advance()  # skip closing quote
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
            # Comments starting with '--'
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
            # Multi-character operators:
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
            # Added support for '[' , ']', and '#' (as length operator) and '-' as operator.
            if self.current_char in "+-*/=(),<>{}.%^#[]":
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

# New AST node for nil literal
class Nil(AST):
    def __init__(self):
        pass
    def __repr__(self):
        return "Nil"

# New AST node for repeat-until loop
class RepeatUntil(AST):
    def __init__(self, block, condition):
        self.block = block  # list of statements
        self.condition = condition

# New AST node for table indexing using square brackets
class TableIndex(AST):
    def __init__(self, table_expr, index_expr):
        self.table_expr = table_expr
        self.index_expr = index_expr

# New AST node for unary operators (such as '-' and '#' for length)
class UnaryOp(AST):
    def __init__(self, op, expr):
        self.op = op  # token containing operator
        self.expr = expr

# ForIn now holds a list of variables.
class Var(AST):
    def __init__(self, name, line=None):
        self.name = name
        self.line = line

class TableLiteral(AST):
    def __init__(self, entries):
        self.entries = entries  # list of (key, value) pairs

class TableAccess(AST):
    def __init__(self, table_expr, field):
        self.table_expr = table_expr
        self.field = field

class Not(AST):
    def __init__(self, expr):
        self.expr = expr

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op  # Token
        self.right = right

# Modified Assignment to allow multiple variables on LHS and multiple expressions on RHS.
class Assignment(AST):
    def __init__(self, left, expr, local=False):
        # left: list of Var, TableAccess, or TableIndex nodes
        # expr: list of expression nodes
        self.left = left
        self.expr = expr
        self.local = local

class Print(AST):
    def __init__(self, expr):
        self.expr = expr

# Modified If node; elseif clauses are built as nested If nodes.
class If(AST):
    def __init__(self, cond, then_block, else_block=None):
        self.cond = cond
        self.then_block = then_block  # list of statements
        self.else_block = else_block  # list of statements or nested If

class While(AST):
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

# For-in loop supports multiple variables.
class ForIn(AST):
    def __init__(self, vars, expr, block):
        self.vars = vars  # list of Var
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
            if self.current_token.type == TT_OP and self.current_token.value == '=':
                self.pos = start_pos
                self.current_token = self.tokens[self.pos]
                return self.assignment()
            else:
                return expr_node
        self.error(f"Unexpected token: {self.current_token}")

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

    # New helper: parse a comma-separated list of variables
    def varlist(self):
        vars = [self.postfix_expr()]
        while self.current_token.type == TT_OP and self.current_token.value == ',':
            self.eat(TT_OP, ',')
            vars.append(self.postfix_expr())
        return vars

    # New helper: parse a comma-separated list of expressions
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
        self.eat(TT_OP, '=')
        expr = self.explist()
        return Assignment(left, expr, local=False)

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
            if self.current_token.type == TT_OP and self.current_token.value == '.':
                self.eat(TT_OP, '.')
                field = self.current_token.value
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

    # New production for unary expressions.
    # For Lua, unary minus (-) has lower precedence than exponentiation,
    # so when '-' is encountered, we call power_expr() to allow correct associativity.
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

    # New production for exponentiation expressions.
    # Exponentiation (^) is right-associative.
    def power_expr(self):
        node = self.postfix_expr()
        if self.current_token.type == TT_OP and self.current_token.value == '^':
            op = self.current_token
            self.eat(TT_OP, '^')
            right = self.unary_expr()  # right-associative
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
        # Handle nil literal
        elif tok.type == TT_KEYWORD and tok.value == 'nil':
            self.eat(TT_KEYWORD, 'nil')
            return Nil()
        elif tok.type == TT_KEYWORD and tok.value == 'not':
            self.eat(TT_KEYWORD, 'not')
            expr = self.primary_expr()
            return Not(expr)
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
            # Support for key in brackets: [expr] = value
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
    # Numbers, Strings, Booleans: nothing to check.

#######################
# Built-in functions for additional Lua features
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

def table_insert(tbl, value, pos=None):
    if not isinstance(tbl, dict):
        raise Exception("table.insert expects a table")
    if pos is not None:
        if isinstance(pos, float) and pos.is_integer():
            pos = int(pos)
        elif not isinstance(pos, int):
            raise Exception("table.insert position must be an integer")
    if pos is None:
        indices = [k for k in tbl.keys() if isinstance(k, int)]
        pos = max(indices) + 1 if indices else 1
        tbl[pos] = value
    else:
        indices = [k for k in tbl.keys() if isinstance(k, int)]
        max_index = max(indices) if indices else 0
        for i in range(max_index, pos - 1, -1):
            tbl[i+1] = tbl.get(i)
        tbl[pos] = value


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
        tbl[i] = tbl.get(i+1)
    if max_index in tbl:
        del tbl[max_index]
    return value

def table_concat(tbl, sep=""):
    if not isinstance(tbl, dict):
        raise Exception("table.concat expects a table")
    result = ""
    i = 1
    while i in tbl:
        result += str(tbl[i])
        i += 1
        if i in tbl:
            result += sep
    return result

def string_len(s):
    if not isinstance(s, str):
        raise Exception("string.len expects a string")
    return len(s)

def string_sub(s, i, j=None):
    if not isinstance(s, str):
        raise Exception("string.sub expects a string")
    if j is None:
        return s[i-1:]
    else:
        return s[i-1:j]

def string_upper(s):
    if not isinstance(s, str):
        raise Exception("string.upper expects a string")
    return s.upper()

def string_lower(s):
    if not isinstance(s, str):
        raise Exception("string.lower expects a string")
    return s.lower()

# Built-in select: returns the count or a list of varargs starting from a given index (Lua indices start at 1)
def builtin_select(first, *args):
    if first == "#":
        return len(args)
    else:
        try:
            n = int(first)
        except:
            raise Exception("select expects a number or '#' as the first argument")
        if n < 1 or n > len(args) + 1:
            return ()  # In Lua, if n is out of bounds, it returns nothing
        return args[n-1:]

# Additional built-in: type, tostring, tonumber
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
# Interpreter and Built-in Functions
#######################
class Interpreter:
    def __init__(self, tree, output_callback=print):
        self.tree = tree
        self.env = {}
        self.output_callback = output_callback
        self.should_stop = False
        # Built-in functions and libraries
        self.env['wait'] = self.builtin_wait
        self.env['require'] = self.builtin_require
        self.env['pairs'] = self.builtin_pairs
        self.env['select'] = builtin_select
        self.env['type'] = builtin_type
        self.env['tostring'] = builtin_tostring
        self.env['tonumber'] = builtin_tonumber
        self.env['math'] = {
            'abs': abs,
            'acos': math.acos,
            'asin': math.asin,
            'atan': math.atan,
            'ceil': math.ceil,
            'cos': math.cos,
            'cosh': math.cosh,
            'deg': math.degrees,
            'exp': math.exp,
            'floor': math.floor,
            'fmod': math.fmod,
            'log': math.log,
            'max': max,
            'min': min,
            'pi': math.pi,
            'rad': math.radians,
            'sin': math.sin,
            'sinh': math.sinh,
            'sqrt': math.sqrt,
            'tan': math.tan,
            'tanh': math.tanh,
            'random': math_random,
            'randomseed': math_randomseed
        }
        self.env['table'] = {
            'insert': table_insert,
            'remove': table_remove,
            'concat': table_concat
        }
        self.env['string'] = {
            'len': string_len,
            'sub': string_sub,
            'upper': string_upper,
            'lower': string_lower
        }

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
        self.visit(self.tree)

    def visit(self, node):
        if self.should_stop:
            self.error("Execution stopped.")
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
        if field in table:
            return table[field]
        self.error(f"Field '{field}' not found in table")

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
        # Fill missing values with nil (None) if there are fewer expressions than variables
        if len(values) < len(node.left):
            values.extend([None] * (len(node.left) - len(values)))
        # If there are more expressions than variables, ignore extras
        for var_node, value in zip(node.left, values):
            if isinstance(var_node, Var):
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

        self.completion_words = list(KEYWORDS) + ['wait', 'require', 'pairs', 'select', 'type', 'tostring', 'tonumber', 'math', 'table', 'string']
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
            "-- UPDATE 2.5.0 --\n"
            "-- This is a comment using '--' as in real Lua\n"
            "local e = { name = \"hero\", attack = function(self) -- Performs an attack\n"
            "    print(self.name .. \" attacks!\")\n"
            "    return 10\n"
            "end }\n\n"
            "function double(x) -- Doubles the input value\n"
            "    return x * 2\n"
            "end\n\n"
            "-- Table member function definition\n"
            "-- function TestModule.doTask(player) -- Executes a task for the player\n"
            "--    print(\"Task executed\")\n"
            "-- end\n\n"
            "if e.name == \"hero\" then\n"
            "    print(\"Welcome, \" .. e.name)\n"
            "elseif e.name == \"villain\" then\n"
            "    print(\"Beware of \" .. e.name)\n"
            "else\n"
            "    print(\"Unknown character\")\n"
            "end\n\n"
            "local damage = e.attack(e)\n"
            "print(\"Damage dealt: \" .. double(damage))\n\n"
            "-- Multiple assignment example:\n"
            "local x, y = 1, 2\n"
            "print(\"x: \" .. x .. \", y: \" .. y)\n\n"
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
            "    print(key .. ' : '  .. value)\n"
            "end\n\n"
            "-- Repeat-until loop example:\n"
            "local count = 0\n"
            "repeat\n"
            "    count = count + 1\n"
            "    print(\"Count: \" .. count)\n"
            "until count >= 3\n\n"
            "-- Math library examples:\n"
            "print(\"Square root of 16 is \" .. math.sqrt(16))\n"
            "print(\"Value of pi is \" .. math.pi)\n"
            "print(\"Random number between 1 and 100: \" .. math.random(1,100))\n\n"
            "-- Table library examples:\n"
            "local arr = {\"a\", \"b\", \"c\"}\n"
            "table.insert(arr, \"d\")\n"
            "print(\"After insert: \" .. table.concat(arr, \",\"))\n"
            "local removed = table.remove(arr, 2)\n"
            "print(\"Removed element: \" .. removed)\n"
            "print(\"After remove: \" .. table.concat(arr, \",\"))\n\n"
            "-- String library examples:\n"
            "print(\"Length of 'hello' is \" .. string.len(\"hello\"))\n"
            "print(\"Uppercase: \" .. string.upper(\"hello\"))\n"
            "print(\"Substring: \" .. string.sub(\"hello\", 2, 4))\n\n"
            "-- Table indexing examples:\n"
            "local t = { a = 1, b = 2, [\"c\"] = 3 }\n"
            "print(\"Value at key 'c': \" .. t[\"c\"])\n"
            "t[\"d\"] = 4\n"
            "print(\"Value at key 'd': \" .. t[\"d\"])\n\n"
            "-- Negative number and exponentiation example:\n"
            "print(\"-2^2 should be -4: \" .. -2^2)  -- parsed as -(2^2) prints -4\n"
            "print(\"(-2)^2 should be 4: \" .. (-2)^2)  -- prints 4\n\n"
            "-- Operators examples:\n"
            "-- Exponentiation: local exp = 2 ^ 3\n"
            "-- Floor division: local flDiv = 10 // 3\n"
            "-- Modulo: local mod = 10 % 3\n"
            "-- Logical: if damage >= 10 and damage ~= 15 then print(\"High damage\") end\n"
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
        self.editor.tag_configure("keyword", foreground="#C678DD")
        self.editor.tag_configure("string", foreground="#98C379")
        self.editor.tag_configure("number", foreground="#D19A66")
        self.editor.tag_configure("boolean", foreground="#E5C07B")
        self.editor.tag_configure("comment", foreground="#7F848E", font=("Consolas", 14, "italic"))
        self.editor.tag_configure("operator", foreground="#56B6C2")
        self.editor.tag_configure("error", underline=True, foreground="#FF5555")
        self.editor.tag_configure("undefined", underline=True, foreground="#61AFEF")

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
                static_check(tree, {"wait": True, "require": True, "pairs": True, "select": True, "type": True, "tostring": True, "tonumber": True, "math": True, "table": True, "string": True})
                self.current_interpreter = Interpreter(tree, output_callback=self.append_output)
                self.current_interpreter.should_stop = False
                self.current_interpreter.env['require'] = self.current_interpreter.builtin_require
                self.current_interpreter.run()
            except Exception as e:
                self.append_output("Error: " + str(e))
        self.run_thread = threading.Thread(target=run_interpreter)
        self.run_thread.start()

    def stop_code(self):
        if self.current_interpreter:
            self.current_interpreter.should_stop = True

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
        for kw in KEYWORDS:
            start = "1.0"
            while True:
                pos = self.editor.search(r'\b' + kw + r'\b', start, stopindex=tk.END, regexp=True)
                if not pos:
                    break
                end = f"{pos}+{len(kw)}c"
                self.editor.tag_add("keyword", pos, end)
                start = end
        for match in re.finditer(r'\b(true|false|nil)\b', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("boolean", start_index, end_index)
        for match in re.finditer(r'(\".*?\"|\'.*?\')', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("string", start_index, end_index)
        for match in re.finditer(r'\b\d+(\.\d+)?\b', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("number", start_index, end_index)
        for match in re.finditer(r'--.*', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("comment", start_index, end_index)
        for match in re.finditer(r'(\+|\-|\*|\/|==|~=|<=|>=|<|>|\^|%|\/\/|\.\.)', content):
            start_index = self.index_from_pos(match.start())
            end_index = self.index_from_pos(match.end())
            self.editor.tag_add("operator", start_index, end_index)

    def index_from_pos(self, pos):
        return "1.0+%dc" % pos

    def show_autocomplete(self):
        pos = self.editor.index(tk.INSERT)
        line_text = self.editor.get("insert linestart", "insert lineend")
        m = re.search(r'(\w+)\.(\w*)$', line_text)
        if m:
            mod_name = m.group(1)
            prefix = m.group(2)
            completions = []
            if self.current_interpreter and mod_name in self.current_interpreter.env:
                mod_obj = self.current_interpreter.env[mod_name]
                if isinstance(mod_obj, dict):
                    doc_dict = mod_obj.get("__doc__", {})
                    for key in mod_obj:
                        if key == "__doc__":
                            continue
                        if key.startswith(prefix):
                            display = key
                            if key in doc_dict:
                                display += " - " + doc_dict[key]
                            completions.append(display)
            if completions:
                self.show_completion_box(completions, pos)
                return
        prefix_match = re.findall(r'(\w+)$', line_text)
        prefix = prefix_match[0] if prefix_match else ""
        completions = [w for w in self.completion_words if w.startswith(prefix)]
        if completions:
            self.show_completion_box(completions, pos)
        else:
            self.hide_completion_box()

    def show_completion_box(self, completions, pos):
        if self.completion_box:
            self.completion_box.destroy()
        self.completion_box = tk.Listbox(self, height=len(completions), font=("Consolas", 12))
        for word in completions:
            self.completion_box.insert(tk.END, word)
        self.completion_box.bind("<<ListboxSelect>>", self.complete_word)
        bbox = self.editor.bbox(tk.INSERT)
        if bbox:
            x, y, width, height = bbox
            abs_x = self.editor.winfo_rootx() + x
            abs_y = self.editor.winfo_rooty() + y + height
            self.completion_box.place(x=abs_x, y=abs_y)
        else:
            self.completion_box.place(x=100, y=100)

    def hide_completion_box(self):
        if self.completion_box:
            self.completion_box.destroy()
            self.completion_box = None

    def complete_word(self, event):
        if not self.completion_box:
            return
        selection = self.completion_box.get(tk.ACTIVE)
        if " - " in selection:
            selection = selection.split(" - ")[0]
        pos = self.editor.index(tk.INSERT)
        line, char = map(int, pos.split('.'))
        line_text = self.editor.get(f"{line}.0", f"{line}.end")
        new_line_text = re.sub(r'(\w+)$', selection, line_text)
        self.editor.delete(f"{line}.0", f"{line}.end")
        self.editor.insert(f"{line}.0", new_line_text)
        self.hide_completion_box()

    def handle_return(self, event):
        current_index = self.editor.index(tk.INSERT)
        line_number = int(current_index.split('.')[0])
        line_start = f"{line_number}.0"
        line_text = self.editor.get(line_start, f"{line_number}.end")
        indent_match = re.match(r'^(\s*)', line_text)
        indent = indent_match.group(1) if indent_match else ""
        if re.search(r'\b(function|while|if|for|repeat)\b', line_text):
            next_line = self.editor.get(f"{line_number+1}.0", f"{line_number+1}.end")
            if next_line.strip() != "end" and next_line.strip() != "until":
                new_indent = indent + "    "
                self.editor.insert(tk.INSERT, "\n" + new_indent + "\n" + indent + "end")
                self.editor.mark_set("insert", f"{line_number + 1}.{len(indent) + 4}")
            else:
                self.editor.insert(tk.INSERT, "\n" + indent)
        else:
            self.editor.insert(tk.INSERT, "\n" + indent)
        return "break"

    def check_syntax(self):
        self.editor.tag_remove("error", "1.0", tk.END)
        self.editor.tag_remove("undefined", "1.0", tk.END)
        code = self.editor.get("1.0", tk.END)
        try:
            lexer = Lexer(code)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            static_check(ast, {"wait": True, "require": True, "pairs": True, "select": True, "type": True, "tostring": True, "tonumber": True, "math": True, "table": True, "string": True})
        except Exception as e:
            err_msg = str(e)
            m = re.search(r'line (\d+)', err_msg)
            if m:
                error_line = m.group(1)
                self.editor.tag_add("error", f"{error_line}.0", f"{error_line}.end")

if __name__ == '__main__':
    app = LuaIDE()
    app.mainloop()
