
# ++++++ Ejercicio catálogo películas +++++++++
import os
#Dominio


class Pelicula:
    def __init__(self, nombre):
        self.__nombre = str(nombre)

    @property
    def nombre(self):
        return self.__nombre

    @nombre.setter
    def nombre(self, nombre):
        self.__nombre = nombre

    def __str__(self):
        return f'Película {self.nombre}'


# Servicio
class CatalogoPeliculas:

    ruta_archivo = 'Catalogo.txt'

    @classmethod
    def agregar_pelicula(cls, pelicula):
        with open(cls.ruta_archivo, 'a', encoding='utf8') as archivo:
            archivo.write(f'{pelicula.nombre}\n')

    @classmethod
    def listar_pelicula(cls):
        with open(cls.ruta_archivo, 'r+', encoding='utf8') as archivo:
            print('Catálogo de Peliculas'.center(50, '-'))
            print(archivo.read())

    @classmethod
    def eliminar(cls):
        os.remove(cls.ruta_archivo)
        print(f'Archivo eliminado: {cls.ruta_archivo}')


# test
# from dominio.Pelicula import Pelidula
# from servicio.catalogo_peliculas import CatalogoPeliculas

mensaje = f"""
Menú:
1. Agregar película
2. Listar películas
3. Eliminar catálogo de películas
4. Salir
"""
opcion = None
while opcion != 4:
    try:
        print(mensaje)
        opcion = int(input('Escribe tu opción (1-4): '))
        """switch = {
            1: CatalogoPeliculas().agregar_pelicula(Pelicula('lol')),
            2: CatalogoPeliculas().listar_pelicula(),
            3: CatalogoPeliculas().eliminar(),
        }
        switch.get(opcion, 'Opción no válida')"""
        if opcion == 1:
            nombre_pelicula = input('Proporciona el nombre de la película: ')
            pelicula = Pelicula(nombre_pelicula)
            CatalogoPeliculas().agregar_pelicula(pelicula)
        elif opcion == 2:
            CatalogoPeliculas().listar_pelicula()
        elif opcion == 3:
            CatalogoPeliculas().eliminar()
    except Exception as e:
        print(f'Ocurrió un error {e}')
        opcion = None


def number_to_string(argument):
    match argument:
        case 0:
            return "zero"
        case 1:
            return "one"
        case 2:
            return "two"
        case default:
            return "something"


number_to_string(4)

# ++++++++  LABORATORIO Mundo PC ##########
# diagrama uml en
# https://www.udemy.com/course/universidad-python-desde-cero-hasta-experto-django-flask-rest-web/learn/lecture/26667098#overview
# _ protegido (#)
# __privado
# orden que contiene computadoras, las cual tiene monitor telcado
# y ratón ( los dos últimos pertenecen a dispositivo de entrada)

class DispositivoEntrada:  # RESPONSABILIDAD crear objetos de tipo DispositivoEntrada
    def __init__(self, tipo_entrada, marca):
        self._tipo_entrada = tipo_entrada
        self._marca = marca

    def __str__(self):
        return f'Marca: {self.marca}, Tipo de entrada: {self.tipo_entrada} '

    @property
    def tipo_entrada(self):
        return self._tipo_entrada

    @property
    def marca(self):
        return self._marca

    @tipo_entrada.setter
    def tipoEntrada(self, tipo_entrada):
        self._tipo_entrada = tipo_entrada

    @marca.setter
    def marca(self, marca):
        self._marca = marca


class Raton(DispositivoEntrada):  # RESPONSABILIDAD crear objetos de tipo ratón
    __contador_raton = 0

    def __init__(self, tipo, marca):
        super().__init__(tipo, marca)
        self.__id_raton = Raton.contador_raton()

    @classmethod
    def contador_raton(cls):
        cls.__contador_raton += 1
        return cls.__contador_raton

    @property
    def id_raton(self):
        return self.__id_raton

    def __str__(self):
        return f'Ratón: ID {self.id_raton}, {super().__str__()} '


class Teclado(DispositivoEntrada):
    __contador_teclado = 0

    def __init__(self, tipo, marca):
        super().__init__(tipo, marca)
        self.__id_teclado = Teclado.contador_teclado()

    @classmethod
    def contador_teclado(cls):
        cls.__contador_teclado += 1
        return cls.__contador_teclado

    @property
    def id_teclado(self):
        return self.__id_teclado

    def __str__(self):
        return f'Teclado: ID {self.id_teclado}, {super().__str__()}'


class Monitor:  # RESPONSABILIDAD crear objetos de tipo Monitor
    __contador_monitores = 0

    def __init__(self, marca, tamaño):
        self.__id_monitor = Monitor.contador_monitores()
        self.__marca = marca
        self.__tamaño = tamaño

    @classmethod
    def contador_monitores(cls):
        cls.__contador_monitores += 1
        return cls.__contador_monitores

    @property
    def marca(self):
        return self.__marca

    @property
    def tamaño(self):
        return self.__tamaño

    @property
    def id_monitor(self):
        return self.__id_monitor

    @marca.setter
    def marca(self, marca):
        self.__marca = marca

    @tamaño.setter
    def tamaño(self, tamaño):
        self.tamaño = tamaño

    def __str__(self):
        return f'Monitor: ID {self.id_monitor}, de marca: {self.marca}  y {self.tamaño} pulgadas'


class Ordenador:
    __contador_ordenador = 0

    def __init__(self, nombre, monitor, teclado, raton):
        self.__id_ordenador = Ordenador.contador_ordenador()
        self.__nombre = nombre
        self.__monitor = monitor
        self.__teclado = teclado
        self.__raton = raton

    @classmethod
    def contador_ordenador(cls):
        cls.__contador_ordenador += 1
        return cls.__contador_ordenador

    @property
    def id_ordenador(self):
        return self.__id_ordenador

    @property
    def nombre(self):
        return self.__nombre

    @property
    def monitor(self):
        return self.__monitor

    @property
    def teclado(self):
        return self.__teclado

    @property
    def raton(self):
        return self.__raton

    @nombre.setter
    def nombre(self, nombre):
        self.__nombre = nombre

    @teclado.setter
    def teclado(self, teclado):
        self.__teclado = teclado

    @raton.setter
    def raton(self, raton):
        self.__raton = raton

    def __str__(self):
        return f'''Ordenador {self.id_ordenador}
    Nombre: {self.nombre}
    Monitor: {self.monitor}
    Teclado: {self.teclado}
    Ratón: {self.raton}
'''

    def imprime_ordenador(self):
        return self.__str__()


class Orden:
    __contador_ordenes = 0

    def __init__(self, ordenadores):
        self.__id_orden = Orden.contador_ordendes()
        self.__ordenadores = list(ordenadores)

    @classmethod
    def contador_ordendes(cls):
        cls.__contador_ordenes += 1
        return cls.__contador_ordenes

    @property
    def id_orden(self):
        return self.__id_orden

    @property
    def ordenadores(self):
        return self.__ordenadores

    @ordenadores.setter
    def ordenadores(self, ordenadores):
        self.__ordenadores = ordenadores

    def agregar_ordenador(self, ordenador):
        self.__ordenadores.append(ordenador)

    def str_ordenadores(self):
        lista = ''
        for ordenador in self.__ordenadores:
            lista += ordenador.imprime_ordenador() + '\n'
        return lista

    def __str__(self):
        return f'Orden {self.id_orden}\n{self.str_ordenadores()}'


raton1 = Raton('USB', 'Racer')
telcado1 = Teclado('Calbe', 'Razer')
monitor1 = Monitor('HP', 27)
ordenador1 = Ordenador('Dell', monitor1, telcado1, raton1)
raton2 = Raton('Bluetooth', 'Racer')
telcado2 = Teclado('Bluetooth', 'Razer')
monitor2 = Monitor('Samsung', 32)
ordenador2 = Ordenador('Mac', monitor2, telcado2, raton2)
raton3 = Raton('Bluetooth', 'Apple')
telcado3 = Teclado('Bluetooth', 'Apple')
monitor3 = Monitor('Integrado', 16)
ordenador3 = Ordenador('Macboockpro', monitor2, telcado2, raton2)

lista_ordandores = [ordenador1, ordenador2]
orden1 = Orden(lista_ordandores)
orden1.agregar_ordenador(ordenador3)
print(orden1)


# Refactorizando ejercicio
class OrdenRefactor:
    contadorOrdenes = 0

    def __init__(self, *ordenadores):
        self.__idOrden = OrdenRefactor.contarOrden()
        self.__ordenadores = list(ordenadores)

    @property
    def idOrden(self):
        return self.__idOrden

    @property
    def ordenadores(self):
        return self.__ordenadores

    @classmethod
    def contarOrden(cls):
        cls.contadorOrdenes += 1
        return cls.contadorOrdenes

    def agregarOrdenador(self, ordenador):
        self.ordenadores.add(ordenador)

    def __str__(self):
        salida = f'Orden {self.idOrden}'
        for ordenador in self.ordenadores:
            salida = salida + '\n' + f'{ordenador}'
        return salida


ordA = Ordenador('HP', Monitor('thosiba', '25"'), Teclado('Bluetooth', 'HP'), Raton('Bluetooth', 'Razer'))
ordB = Ordenador('MACBOOK', Monitor('Apple pro Res', '16"'), Teclado('Bluetooth', 'apple'), Raton('bluetooth', 'Razer'))
print(OrdenRefactor(ordA, ordB))

# ++++++++ desafío Edube 1: 3 en raya +++++++++++++
"""
la computadora (es decir, su programa) debe jugar el juego usando 'X'; el usuario (p. ej., usted) debe jugar el juego 
usando 'O'; el primer movimiento pertenece a la computadora: siempre pone su primera 'X' en el medio del tablero;
todos los cuadrados están numerados fila por fila comenzando con 1 (consulte la sesión de ejemplo a continuación como 
referencia) el usuario ingresa su movimiento ingresando el número del cuadrado que elige; el número debe ser válido, es decir,
debe ser un número entero, debe ser mayor que 0 y menor que 10, y no puede apuntar a un campo que ya está ocupado;
el programa verifica si el juego ha terminado: hay cuatro veredictos posibles: el juego debe continuar, el juego termina 
en empate, tú ganas o la computadora gana; la computadora responde con su jugada y se repite la verificación;
no implemente ninguna forma de inteligencia artificial: una elección de campo aleatoria hecha por la computadora es lo 
suficientemente buena para el juego. La sesión de ejemplo con el programa puede tener el siguiente aspecto:
Enter your move: 1
+-------+-------+-------+
|       |       |       |
|   O   |   2   |   3   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   4   |   X   |   6   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   7   |   8   |   9   |
|       |       |       |
+-------+-------+-------+
+-------+-------+-------+
|       |       |       |
|   O   |   X   |   3   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   4   |   X   |   6   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   7   |   8   |   9   |
|       |       |       |
+-------+-------+-------+
... playing ...
Enter your move: 7
+-------+-------+-------+
|       |       |       |
|   O   |   X   |   X   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   O   |   X   |   X   |
|       |       |       |
+-------+-------+-------+
|       |       |       |
|   O   |   O   |   9   |
|       |       |       |
+-------+-------+-------+
You won!

Requisitos
Implemente las siguientes características:

el tablero debe almacenarse como una lista de tres elementos, mientras que cada elemento es otra lista de tres elementos
(las listas internas representan filas) para que se pueda acceder a todos los cuadrados usando la siguiente sintaxis:

tablero[fila][columna]
cada uno de los elementos de la lista interna puede contener 'O', 'X' o un dígito que represente el número del cuadrado
 (tal cuadrado se considera libre)la apariencia del tablero debe ser exactamente igual a la presentada en el ejemplo.
implementar las funciones definidas para usted en el editor.

Se puede dibujar un número entero aleatorio utilizando una función de Python llamada randrange().
El programa de ejemplo a continuación muestra cómo usarlo (el programa imprime diez números aleatorios del 0 al 8).
Nota: la instrucción from-import proporciona acceso a la función randrange definida
dentro de un módulo externo de Python llamado random.

from random import randrange

for i in range(10):
    print(randrange(8))
"""
from random import randrange


def display_board(board):
    # The function accepts one parameter containing the board's current status
    # and prints out a "TABLE" of a matrix[x][y] to the console.
    length = len(board[0])
    middle = ("/" + '\t\t') * (length + 1)
    begend = "+-------" * length + "+"

    for fila in board:
        if fila == board[0]:
            print(begend)
        print(middle)
        pos = 0
        for elemento in fila:
            if pos == 0:
                print("/", elemento, "/", sep="\t", end="")
            else:
                print("", elemento, "/", sep="\t", end="")
            if pos == len(fila) - 1:
                print()
                print(middle)
                print(begend)
            pos += 1


def enter_move(board):
    # The function accepts the board's current status, asks the user about their move,
    # checks the input, and updates the board according to the user's decision.
    correct_move = False
    while not correct_move:
        try:
            move = int(input("Enter your move: "))
            board = moving(board,'O',move)
            correct_move = True
        except TypeError:
            print(f'move not allowed, only integers')
        except Exception as e:
            print(e)
    return board


def make_list_of_free_fields(board):
    # The function browses the board and builds a list of all the free squares;
    # the list consists of tuples, while each tuple is a pair of row and column numbers.
    free = []
    for i_b, f in enumerate(board):
        for i_f, pos in enumerate(f):
            if pos not in ('X', 'O'):
                free.append((i_b, i_f))
    return free


def victory_for(board, sign):
    # The function analyzes the board's status in order to check if
    # the player using 'O's or 'X's has won the game
    end = False
    for m in ['123', '456', '789', '147', '258', '369', '159', '357']:
        if (board[movements.get(int(m[0]))[0]][movements.get(int(m[0]))[1]] == sign
        and board[movements.get(int(m[1]))[0]][movements.get(int(m[1]))[1]] == sign
        and board[movements.get(int(m[2]))[0]][movements.get(int(m[2]))[1]]) == sign:
            end = True
    if end:
        if sign == 'O':
            print('You won!')
        else:
            print('You loose!')
    return end


def draw_move(board, ini=False):
    # The function draws the computer's move and updates the board.
    correct_move = False
    while not correct_move:
        try:
            if ini == True:
                move = 5
            else:
                move = randrange(10)
            board = moving(board, 'X', move)
            correct_move = True
        except TypeError:
            print(f'move not allowed, only integers')
        except Exception as e:
            print(e)
    return board


def moving(board, value, move=None):
    if not move:
        move = randrange(10)
    if not movements.get(move) in make_list_of_free_fields(board):
        raise CustomException('Movement already used')
    else:
        board[:] = [[value if pos == move else pos for pos in f] for f in board]
    display_board(board)
    return board


def init_board(f=3, c=3):
    return [[x + (y * f) for x in range(1, f + 1)] for y in range(c)]


class CustomException(Exception):
    def __init__(self, mensaje):
        self.message = mensaje


if __name__ == '__main__':
    movements ={
                1: (0, 0),
                2: (0, 1),
                3: (0, 2),
                4: (1, 0),
                5: (1, 1),
                6: (1, 2),
                7: (2, 0),
                8: (2, 1),
                9: (2, 2)}
    board = init_board()
    draw_move(board, True)
    end = False
    while not make_list_of_free_fields(board) == []:
        board = enter_move(board)
        end = victory_for(board, 'O')
        if end:
            break
        draw_move(board)
        end = victory_for(board, 'O')
        if end:
            break
    if not end:
        print(f'Tie!')



# +++++++ DESAFIO SNAKE +++++++
class SnakeException(Exception):
    def __init__(self, mensaje):
        self.message = mensaje

# IMPORTS
import itertools as it
#import lib.SnakeException as SnakeException
import math, re

#   CLASS
class SnakeManager:
    #   CLASS VARIABLES
    __numberOfPaths = -1
    __paths = {
        'L': -1,
        'R': 1,
        'D': 1,
        'U': -1,
    }

    def __init__(self, board, snake, depth):
        self.__board = self.initBoard(board)
        self.__snake = self.initSnake(snake)
        self.__depth = self.initdepth(depth)
        self.__snakeOnBorad = self.setSnakeOnBoard()
        self.__tamBoard = board

    @property
    def board(self):
        return self.__board

    @board.setter
    def board(self, board):
        self.__board = board

    @property
    def snake(self):
        return self.__snake

    @snake.setter
    def snake(self, snake):
        self.__snake = snake

    @property
    def depth(self):
        return self.__depth

    @depth.setter
    def depth(self, depth):
        self.__depth = depth

    @property
    def snakeOnBorad(self):
        return self.__snakeOnBorad

    @snakeOnBorad.setter
    def snakeOnBorad(self, snakeOnBorad):
        self.__snakeOnBorad = snakeOnBorad

    @property
    def tamBboard(self):
        return self.__tamBoard

    @classmethod
    def initBoard(cls, board):
        if not 1 <= board[0] <= 10:
            raise SnakeException('The board rows isn\'t between 1 and 10')
        elif not len(board) == 2:
            raise SnakeException('The board length must be 2')
        return board

    def initSnake(self, snake):
        if not 1 <= len(snake) <= 7:
            raise SnakeException('The snake rows isn\'t between 3 and 7')
        elif not len(snake[0]) == 2:
            raise SnakeException('The snake[i] length must be 2')
        return snake

    @classmethod
    def initdepth(cls, depth):
        if not 1 <= depth <= 20:
            raise SnakeException('The depth is not correct')
        return depth

    def cut(self, elem):
        pass

    def setSnakeOnBoard(self, popped=[]):
        board_tmp = [[0 for x in range(self.board[0])] for y in range(self.board[1])]
        #   if the snake is moving
        if self.__numberOfPaths != -1:
            board_tmp2 = self.snakeOnBorad
            # Erase last possition
            if popped != []:
                board_tmp2[popped[1]][popped[0]] = 0
            if board_tmp2[self.snake[0][1]][self.snake[0][0]] == 1:
                raise SnakeException('Intersection  at the movement')
        #   Snake array tour
        i = 0
        for pos in self.snake:
            # Checking that the snake fits in the table
            if i == 0:
                incr = 8
            else:
                incr = 1
            if not pos[0] < self.board[0] or not pos[1] < self.board[1]:
                raise SnakeException(f'The snake don\'t fit on board {self.board} in {pos}'
                                     f'\n does not comply with the Guaranteed Restriction 0 ≤ snake[i][j]< board[j] ')
            # We go through the rows
            for row in range(self.board[0]):
                # We go through the colums
                for colum in range(self.board[1]):
                    # If the current position matches the position of the snake
                    if [row, colum] == [pos[0], pos[1]]:
                        if board_tmp[colum][row] != 0:
                            raise SnakeException('self-intersections')
                        board_tmp[colum][row] = incr
            i +=1
        self.__numberOfPaths += 1
        return board_tmp

    def moveSnakeOnBoard(self, path):
        popped = []
        if path in ('R', 'L'):
            incr = self.__paths.get(path)
            # print(incr)
            if self.snake[0][0] + incr < 0:
                raise SnakeException('The snake goes off the board at the movement')
            self.snake.insert(0, [self.snake[0][0] + incr, self.snake[0][1]])
            popped = [self.snake[-1][0], self.snake[-1][1]]
            self.snake.pop()
        elif path in ('D', 'U'):
            incr = self.__paths.get(path)
            # print(incr)
            if self.snake[0][1] + incr < 0:
                raise SnakeException('The snake goes off the board at the movement')
            self.snake.insert(0, [self.snake[0][0], self.snake[0][1] + incr])
            popped = [self.snake[-1][0], self.snake[-1][1]]
            self.snake.pop()
        # print(self.snake)
        self.snakeOnBorad = self.setSnakeOnBoard(popped)

    def isPossible(self):
        possible = possible = list(it.product('LURD', repeat=self.depth))
        partL = possible[:math.ceil(len(possible)*0.25)]
        partU = possible[math.floor(len(possible)*0.25):math.ceil(len(possible)*0.50)]
        partR = possible[math.floor(len(possible)*0.50):math.ceil(len(possible)*0.75)]
        partD = possible[math.floor(len(possible)*0.75):len(possible)]
        validPath = set()

        condition = str(self.snake[0][0])+str(self.snake[0][1])
        match condition:
            # Edge
            case condition if condition == '00':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partR
                else:
                    possible = partR + partD

            case condition if condition == f'{self.tamBboard[1]-1}0':
                if [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partL
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partD
                else:
                    possible = partL + partD

            case condition if condition == f'0{self.tamBboard[1]-1}':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partR
                else:
                    possible = partR + partD

            case condition if condition == f'{self.tamBboard[1]-1}{self.tamBboard[0]-1}':
                if [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partD
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partU
                else:
                    possible = partL + partU

            case condition if condition[0] == '0' and condition != '00':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partL + partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partL + partR
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partR + partD
                else:
                    possible = partL + partR + partD

            case condition if condition[1] == '0' and condition != '00':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partU + partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partU + partR
                elif [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partR + partD
                else:
                    possible = partU + partR + partD

            case condition if condition[0] == f'{self.tamBboard[0]-1}' and condition != f'{self.tamBboard[1]-1}{self.tamBboard[0]-1}':
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partL + partU
                elif [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partL + partR
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partU + partR
                else:
                    possible = partL + partU + partR

            case condition if condition[1] == f'{self.tamBboard[1]-1}' and condition != f'{self.tamBboard[1]-1}{self.tamBboard[0]-1}':
                if [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partL + partU
                elif [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partL + partD
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partU + partD
                else:
                    possible = partL + partU + partD
            # Snake body
            case condition if condition not in ('00', f'0{self.tamBboard[1]-1}', f'{self.tamBboard[1]-1}0', f'{self.tamBboard[1]-1}{self.tamBboard[0]-1}'):
                if [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] + 1) in s] != []:
                    # R OFF
                    possible = partL + partU + partD
                elif [s for s in self.snake if str(self.snake[0][0] + 1) + str(self.snake[0][1]) in s] != []:
                    # D OFF
                    possible = partL + partU + partR
                elif [s for s in self.snake if str(self.snake[0][0] - 1) + str(self.snake[0][1]) in s] != []:
                    # U OFF
                    possible = partL + partR + partD
                elif [s for s in self.snake if str(self.snake[0][0]) + str(self.snake[0][1] - 1) in s] != []:
                    # L OFF
                    possible = partU + partR + partD



        return possible

    def searchL(self):
        validPath = set()
        possible = self.isPossible()
        for p in possible:
            status, path, loop = self.boundedDepthSearch(p)
            if status == 'ok':
                validPath.add(str(p))
        return len(validPath)

    def boundedDepthSearch(self, loopPath):
        photoBoard = self.snakeOnBorad
        photoSnake = self.photoSnake(self.snake)
        curr = []
        loop = 0
        for p in loopPath:
            try:
                self.moveSnakeOnBoard(p)
                curr.append(p)
                if len(curr) == self.depth:
                    status = 'ok'
            except:
                status = 'ko'
        loop = len(curr)-1
        self.snakeOnBorad = photoBoard
        self.snake = photoSnake
        return status, curr, loop

    def photoSnake(self, state):
        tmp = []
        for s in state:
            tmp.append(s)
        return tmp

    def __str__(self):
        out = ''
        for row in pr.snakeOnBorad:
            out = out + row + '\n'
        return out


if __name__ == '__main__':
    # pr = SnakeManager([4, 3], [[2, 2], [3, 2], [3, 1], [3, 0], [2, 0], [1, 0], [0, 0]], 3)
    pr = SnakeManager([10, 10], [[5,5], [5,4], [4,4], [4,5]], 4)
    # pr = SnakeManager([2, 3], [[0,2], [0,1], [0,0], [1,0], [1,1], [1,2]], 10)
    print(pr.searchL())
