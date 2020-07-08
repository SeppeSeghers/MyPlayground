from google_trans import Translator

translator = Translator()


def silly_trans(txt, silly, destin):
    detected = translator.detect(txt)
    print("Detected ", detected.lang, " with ", detected.confidence, " confidence.")
    if silly == 'n':
        print('-> ', translator.translate(txt, dest=destin).text)
    else:
        silly_list = txt.split()
        translated_list = translator.translate(silly_list, dest=destin, src=detected.lang)
        silly_str = ''
        for word in translated_list:
            silly_str += str(word.text) + ' '
        print('-> ', silly_str)


text = input("What do you want to translate?")
silly_way = input("Do you want to translate it in a silly way? (y/n)")
destin = input("Into what language do you want to translate it? (en, nl, ...)")

silly_trans(text, silly_way, destin)
