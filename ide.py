import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
import re
import time
import threading

#######################
# Lexer
#######################
TT_NUMBER = 'NUMBER'
TT_STRING = 'STRING'
TT_IDENT = 'IDENT'
TT_KEYWORD = 'KEYWORD'
TT_OP = 'OP'
TT_EOF = 'EOF'

# Keywords (note: "wait", "break", "require", and now "pairs" are built-ins)
KEYWORDS = {
    'if', 'then', 'else', 'end', 'while', 'do', 'print',
    'local', 'true', 'false', 'for', 'in', 'and', 'or', 'not',
    'function', 'return', 'break'
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
            if self.current_char in " \t\r":
                self.skip_whitespace()
                continue
            if self.current_char == "\n":
                self.advance()
                continue
            if self.current_char == '-' and self.peek() == '-':
                self.advance(); self.advance()
                self.skip_comment()
                continue
            if self.current_char.isdigit():
                return self.number()
            if self.current_char in ('"', "'"):
                return self.string()
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()
            if self.current_char in "+-*/=(),<>{}.%^":
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

class Assignment(AST):
    def __init__(self, left, expr, local=False):
        self.left = left  # Var or TableAccess
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
        while self.current_token.type != TT_EOF and self.current_token.value not in ('else', 'end'):
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
        if self.current_token.type == TT_IDENT or (self.current_token.type == TT_OP and self.current_token.value == '{'):
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

    def local_assignment(self):
        self.eat(TT_KEYWORD, 'local')
        left = self.postfix_expr()
        self.eat(TT_OP, '=')
        expr = self.expr()
        return Assignment(left, expr, local=True)

    def assignment(self):
        left = self.postfix_expr()
        self.eat(TT_OP, '=')
        expr = self.expr()
        return Assignment(left, expr)

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
        else_block = None
        if self.current_token.type == TT_KEYWORD and self.current_token.value == 'else':
            self.eat(TT_KEYWORD, 'else')
            else_block = self.statement_list()
        self.eat(TT_KEYWORD, 'end')
        return If(cond, then_block, else_block)

    def while_statement(self):
        self.eat(TT_KEYWORD, 'while')
        cond = self.expr()
        self.eat(TT_KEYWORD, 'do')
        body = self.statement_list()
        self.eat(TT_KEYWORD, 'end')
        return While(cond, body)

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
                return Assignment(TableAccess(Var(name), field), func_literal)
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

    def exponent_expr(self):
        node = self.postfix_expr()
        if self.current_token.type == TT_OP and self.current_token.value == '^':
            op = self.current_token
            self.eat(TT_OP, '^')
            right = self.exponent_expr()
            node = BinOp(node, op, right)
        return node

    def term(self):
        node = self.exponent_expr()
        while self.current_token.type == TT_OP and self.current_token.value in ('*', '/', '%', '//'):
            op = self.current_token
            self.eat(TT_OP, op.value)
            right = self.exponent_expr()
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
            if (self.current_token.type == TT_IDENT and
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
        static_check(node.expr, env)
        if isinstance(node.left, Var):
            env[node.left.name] = True
        elif isinstance(node.left, TableAccess):
            static_check(node.left, env)
        else:
            static_check(node.left, env)
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
    elif isinstance(node, Not):
        static_check(node.expr, env)
    # Numbers, Strings, Booleans: nothing to check.

#######################
# Interpreter and Built-in Functions
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

class Interpreter:
    def __init__(self, tree, output_callback=print):
        self.tree = tree
        self.env = {}
        self.output_callback = output_callback
        self.should_stop = False
        self.env['wait'] = self.builtin_wait
        self.env['require'] = self.builtin_require
        self.env['pairs'] = self.builtin_pairs

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

    def visit_Not(self, node):
        return not self.visit(node.expr)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op.value
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
        elif op == 'and':
            return left and right
        elif op == 'or':
            return left or right
        else:
            self.error(f"Unknown binary operator {op}")

    def visit_Assignment(self, node):
        val = self.visit(node.expr)
        if isinstance(node.left, Var):
            self.env[node.left.name] = val
        elif isinstance(node.left, TableAccess):
            table = self.visit(node.left.table_expr)
            table[node.left.field] = val
        else:
            self.error("Invalid assignment target")
        return val

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

        self.completion_words = list(KEYWORDS) + ['wait', 'require', 'pairs']
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
            "-- UPDATE 1.0.0 --\n"
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
            "end\n\n"
            "local damage = e.attack(e)\n"
            "print(\"Damage dealt: \" .. double(damage))\n\n"
            "-- Numeric for loop example:\n"
            "for i = 1, 5, 1 do\n"
            "    print(i)\n"
            "end\n\n"
            "-- For-in loop example using pairs:\n"
            "for key, value in pairs({a=1, b=2, c=3}) do\n"
            "    print(key .. ' : '  .. value)\n"
            "end\n\n"
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
        self.editor.tag_configure("keyword", foreground="#61AFEF")
        self.editor.tag_configure("string", foreground="#98C379")
        self.editor.tag_configure("number", foreground="#D19A66")
        self.editor.tag_configure("comment", foreground="#5C6370")
        self.editor.tag_configure("error", underline=True, foreground="red")
        self.editor.tag_configure("undefined", underline=True, foreground="blue")

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
        code = self.editor.get("1.0", tk.END)
        self.output.delete(1.0, tk.END)
        def run_interpreter():
            try:
                lexer = Lexer(code)
                tokens = lexer.tokenize()
                parser = Parser(tokens)
                tree = parser.parse()
                # Added "pairs" to the static check environment.
                static_check(tree, {"wait": True, "require": True, "pairs": True})
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
        self.editor.tag_remove("comment", "1.0", tk.END)
        for kw in KEYWORDS:
            start = "1.0"
            while True:
                pos = self.editor.search(r'\b' + kw + r'\b', start, stopindex=tk.END, regexp=True)
                if not pos:
                    break
                end = f"{pos}+{len(kw)}c"
                self.editor.tag_add("keyword", pos, end)
                start = end
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
        if re.search(r'\b(function|while|if|for)\b', line_text):
            next_line = self.editor.get(f"{line_number+1}.0", f"{line_number+1}.end")
            if next_line.strip() != "end":
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
            static_check(ast, {"wait": True, "require": True, "pairs": True})
        except Exception as e:
            err_msg = str(e)
            m = re.search(r'line (\d+)', err_msg)
            if m:
                error_line = m.group(1)
                self.editor.tag_add("error", f"{error_line}.0", f"{error_line}.end")

if __name__ == '__main__':
    app = LuaIDE()
    app.mainloop()
