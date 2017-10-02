from html.parser import HTMLParser
from html.entities import name2codepoint

class MyHTMLParser(HTMLParser):
    text = []
    def handle_starttag(self, tag, attrs):
        """
        attr: ('id', 'NYT_ENG_20130607.0076')
        attr: ('type', 'story')
        """
        return
        print("Start tag:", tag)
        for attr in attrs:
            print("     attr:", attr)

    def handle_endtag(self, tag):
        return
        print("End tag  :", tag)

    def handle_data(self, data):
        if data.strip():
            #print("Data     :", data)
            self.text.append(data.strip())


    def handle_comment(self, data):
        return
        #print("Comment  :", data)

    def handle_entityref(self, name):
        c = chr(name2codepoint[name])
        #print("Named ent:", c)

    def handle_charref(self, name):
        if name.startswith('x'):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        #print("Num ent  :", c)

    def handle_decl(self, data):
        return
        #print("Decl     :", data)

    def get_text(self):
        return "\n".join(self.text)

#parser = MyHTMLParser()
