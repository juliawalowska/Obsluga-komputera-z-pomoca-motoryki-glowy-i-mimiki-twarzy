import pygame
import random
import math

# Inicjalizacja Pygame
pygame.init()

# Pobieranie informacji o ekranie
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

# Ustawienia okna
width, height = 800, 600
screen = pygame.display.set_mode((800, 600), pygame.FULLSCREEN)

pygame.display.set_caption('Śledzenie szlaczka')

# Kolory
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Rysowanie szlaczka
def draw_pattern(points):
    pygame.draw.lines(screen, black, False, points, 3)

# Obliczanie dokładności
def calculate_accuracy(points, hits):
    return hits / len(points) if points else 0

# Główna pętla gry
def main():
    running = True

    points = []
    start_x = random.randint(100, width - 100)
    start_y = random.randint(100, height - 100)
    points.append((start_x, start_y))
    pygame.mouse.set_pos(start_x, start_y)

    direction = random.choice(['up', 'down', 'left', 'right'])
    while len(points) < 1000:  # Maksymalna liczba punktów
        length = random.randint(150, 300)  # Długość segmentu (minimum 200 pikseli)

        if direction == 'up':
            new_y = max(start_y - length, 100)  # Zapobiega wychodzeniu poza ekran
            for i in range(start_y, new_y, -1):  # Dodajemy krotki (x, y) pionowo w górę
                points.append((start_x, i))
            start_y = new_y
            direction = random.choice(['left', 'right'])
        elif direction == 'down':
            new_y = min(start_y + length, height - 100)  # Zapobiega wychodzeniu poza ekran
            for i in range(start_y, new_y):  # Dodajemy krotki (x, y) pionowo w dół
                points.append((start_x, i))
            start_y = new_y
            direction = random.choice(['left', 'right'])
        elif direction == 'left':
            new_x = max(start_x - length, 100)  # Zapobiega wychodzeniu poza ekran
            for i in range(start_x, new_x, -1):  # Dodajemy krotki (x, y) poziomo w lewo
                points.append((i, start_y))
            start_x = new_x
            direction = random.choice(['up', 'down'])
        elif direction == 'right':
            new_x = min(start_x + length, width - 100)  # Zapobiega wychodzeniu poza ekran
            for i in range(start_x, new_x):  # Dodajemy krotki (x, y) poziomo w prawo
                points.append((i, start_y))
            start_x = new_x
            direction = random.choice(['up', 'down'])

    hits = 0  # Licznik trafień
    checked_indices = set()  # Zestaw odwiedzonych indeksów, aby uniknąć podwójnego liczenia

    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()  # Początkowy czas gry w milisekundach
    end_time = None  # Zmienna do przechowywania czasu zakończenia

    while running:
        screen.fill(white)  # Wypełniamy tło białym kolorem
        draw_pattern(points)  # Rysuj stały szlaczek
        cursor_pos = pygame.mouse.get_pos()  # Pozycja kursora

        # Sprawdzanie, czy kursor jest blisko punktów szlaczka
        for index, point in enumerate(points):
            distance = math.hypot(point[0] - cursor_pos[0], point[1] - cursor_pos[1])
            if distance < 10:  # Promień 10 pikseli
                if index not in checked_indices:  # Liczymy tylko, gdy indeks nie był wcześniej trafiony
                    hits += 1
                    checked_indices.add(index)  # Dodaj indeks do zestawu

        # Sprawdzanie, czy kursor jest blisko ostatniego punktu
        last_point = points[-1]
        distance_to_last_point = math.hypot(last_point[0] - cursor_pos[0], last_point[1] - cursor_pos[1])

        if distance_to_last_point < 10 and end_time is None:  # Gdy kursor jest blisko ostatniego punktu
            end_time = pygame.time.get_ticks()  # Zapisz czas zakończenia

        # Obliczanie i wyświetlanie dokładności
        accuracy = calculate_accuracy(points, hits)
        accuracy_text = f'Dokładność: {accuracy:.2%}'

        # Obliczanie czasu trwania gry (jeśli osiągnięto ostatni punkt, liczymy od startu do end_time)
        if end_time is not None:
            elapsed_time = (end_time - start_time) / 1000  # Czas do momentu zakończenia w sekundach
        else:
            elapsed_time = (pygame.time.get_ticks() - start_time) / 1000  # Aktualny czas trwania gry

        time_text = f'Czas gry: {elapsed_time:.2f} sek.'  # Używamy formatu z dwoma miejscami po przecinku

        # Renderowanie tekstu na ekranie
        font = pygame.font.SysFont(None, 36)
        accuracy_surface = font.render(accuracy_text, True, red)
        time_surface = font.render(time_text, True, red)

        # Wyświetlanie tekstu na ekranie
        screen.blit(accuracy_surface, (10, 10))  # Wyświetl dokładność
        screen.blit(time_surface, (10, 50))  # Wyświetl czas gry

        # Sprawdzanie zdarzeń
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()
