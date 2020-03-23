import pygame as pg
import sys

pg.init() #라이브러리 초기화
ourScreen = pg.display.set_mode((500,700))
pg.display.set_caption('엘리베이터 시각화 해주기!')
finish = False
colorB = True # 색 조건을 참으로 설정
x, y = 10, 600 #기본 위치값 설정
color = (0,255,255)

#------------------엘리베이터 알고리즘 정보--------------------#
nowfloor = 0#+1 #현재 층
way = 'up' #현재 방향
wantgo = 1 #원하는 층
#------------------엘리베이터 알고리즘 정보--------------------#

#------------------디스플레이 구현 과정 및 연산 과정--------------------#
clock = pg.time.Clock()
while not finish:
#--------------------------------------------------------------#
    for event in pg.event.get():
        if event.type == pg.QUIT:
            finish = True
            terminate()
        
        '''
        if event.type == pg.KEYDOWN and event.key == pg.K_SPACE :
            colorB = not colorB
        '''
        
        if colorB:
            color = (0,255,255)
        else:
            color = (0,0,255)
            
#---------------------------------------------------------#
        '''
        pressed = pg.key.get_pressed() # 입력을 한 칸씩 받음
        if pressed[pg.K_UP]: y -= 70 # 윗 방향키
        if pressed[pg.K_DOWN]: y += 70
        if pressed[pg.K_LEFT]: x -= 70
        if pressed[pg.K_RIGHT]: x += 70
        '''
#---------------------------------------------------------#
        
    if way == 'up' :
        y-=70#올라가는 방향이라면 위로 한 층 올려준다.
        nowfloor = nowfloor + 1
        print('NOW:',nowfloor)
    if way == 'down' :
        y+=70 #내려가는 방향은 아래로 한 층 내려준다.
        nowfloor = nowfloor - 1
        print('NOW:',nowfloor)
    if way == 'stop' :
        y=y #그대로!
        nowfloor = nowfloor #그대로!!
        print('NOW:',nowfloor)
        
    ourScreen.fill((0,0,0)) #화면 채우기
    pg.draw.rect(ourScreen,color,[x,y,50,70],10) # 위 모드 변화에 동적으로
                                                    #도형그리기
    pg.display.flip() # 화면 전체를 업데이트
    clock.tick(60)
    press = int(input('pressed? 0 or 1: '))
    if press == 1:
        wantgo = int(input('where does he wanna go?:'))
        if wantgo < nowfloor:
            way = 'down'
        elif wantgo > nowfloor:
            way = 'up'
    
    if nowfloor == wantgo: #원하는 층과 현재 층이 같은 경우 멈춘다.
            way = 'stop'
    
def terminate():
    pg.quit()
    sys.exit()

def getLeftTopOfTile(tileX, tileY):
    left = XMARGIN + (tileX * TILESIZE) + (tileX - 1)
    top = YMARGIN + (tileY * TILESIZE) + (tileY - 1)
    return (left, top)

def drawTile(tilex, tiley, number, adjx=0, adjy=0):
    # draw a tile at board coordinates tilex and tiley, optionally a few
    # pixels over (determined by adjx and adjy)
    left, top = getLeftTopOfTile(tilex, tiley)
    pygame.draw.rect(DISPLAYSURF, TILECOLOR, (left + adjx, top + adjy, TILESIZE, TILESIZE))
    textSurf = BASICFONT.render(str(number), True, TEXTCOLOR)
    textRect = textSurf.get_rect()
    textRect.center = left + int(TILESIZE / 2) + adjx, top + int(TILESIZE / 2) + adjy
    DISPLAYSURF.blit(textSurf, textRect)
