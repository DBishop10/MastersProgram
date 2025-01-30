import pymysql

def SQL_refresh_database(db):
    # delete existing tables
    cursor = db.cursor()
    sql = """
DROP TABLE IF EXISTS Characters, Users, Cards;
    """
    cursor.execute(sql)
    db.commit()
  
    # then recreate the tables
    SQL_create_table_users(cursor)
    SQL_create_table_characters(cursor)
    SQL_create_table_cards(cursor)
    db.commit()
    db.close()
    return 0


def SQL_create_table_cards(cursor):
    sql = """
   CREATE TABLE `NotAClue`.`Cards` (
   `Id` int,
   `name` VARCHAR(45),
   `type` VARCHAR(45));
    """
    cursor.execute(sql)
    return 0


def SQL_add_card(db, id, name, type):
    cursor = db.cursor()
    sql = """
    INSERT INTO `NotAClue`.`Cards`
    (`Id`,
    `name`, 
    `type`)
    VALUES
    (%s, 
    %s,
    %s);
    """
    cursor.execute(sql, (id, name, type))
    db.commit()
    return 0



def SQL_create_table_users(cursor):
    sql = """
   CREATE TABLE `NotAClue`.`Users` (
  `SID` varchar(255),
  `username` VARCHAR(45) NULL,
  PRIMARY KEY (`SID`));
    """
    cursor.execute(sql)
    return 0


def SQL_create_table_characters(cursor):
    sql = """
    CREATE TABLE `NotAClue`.`Characters` (
    `character` VARCHAR(45) NOT NULL,
    `SID` varchar(255),
    PRIMARY KEY (`character`),
    FOREIGN KEY (`SID`)
    REFERENCES `NotAClue`.`Users` (`SID`)
    ON DELETE CASCADE
    ON UPDATE NO ACTION);
        """
    cursor.execute(sql)
    return 0


def SQL_handle_user_join(db, sid, username):
    cursor = db.cursor()
    sql = """
    INSERT INTO `NotAClue`.`Users`
    (`SID`,
    `username`)
    VALUES
    (%s, 
    %s);
    """
    cursor.execute(sql, (sid, username))
    db.commit()
    return 0


def SQL_get_characters_and_usernames(db):
    cursor = db.cursor()
    sql = """
    SELECT a.`character`,
           b.`username`
    FROM `NotAClue`.`Characters` AS  a
		LEFT JOIN `NotAClue`.`Users` AS b 
        ON a.SID = b.SID;
        
    """
    cursor.execute(sql)
    res = cursor.fetchall()
    return res


def SQL_get_username_based_on_sid(db, sid):
    cursor = db.cursor()
    sql = """
    SELECT username
	FROM NotAClue.Users
	WHERE sid=%s;
    """
    cursor.execute(sql,(sid,))
    res = cursor.fetchall()
    if res:
        return res[0][0]
    return None


def SQL_handle_player_select(db, character, sid):
    cursor = db.cursor()
    sql = """
    INSERT INTO `NotAClue`.`Characters`
    (`character`,
    `SID`)
    VALUES
    (%s,
    %s);
        """
    cursor.execute(sql, (character, sid))
    db.commit()
    return 0


def SQL_get_character_based_on_username(db, username):
    cursor = db.cursor()
    sql = """
    SELECT a.`character`
    FROM `NotAClue`.`Characters` AS  a
		LEFT JOIN `NotAClue`.`Users` AS b 
        ON a.SID = b.SID
    WHERE b.username=%s;
            """
    cursor.execute(sql, (username,))
    res = cursor.fetchall()
    if res:
        return res[0][0]
    return None

def SQL_get_card_id(db, cardName):
    cursor = db.cursor()
    sql = """
    SELECT c.`Id`
    FROM `NotAClue`.`Cards` c
    WHERE c.name = %s;
            """
    cursor.execute(sql, (cardName,))
    res = cursor.fetchall()
    if res:
        return res[0][0]
    return None

def SQL_delete_character_based_on_charactername(db, charactername):
    cursor = db.cursor()
    sql = """
    DELETE FROM `NotAClue`.`Characters` as a
    WHERE a.character = %s;
            """
    cursor.execute(sql, (charactername,))
    db.commit()
    return 0

