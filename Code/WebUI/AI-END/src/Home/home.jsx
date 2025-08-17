import Form from "./components/form";
import styles from "./Home.module.css";

const Home = () => {
    const text = "Park_Intelligent_Platform";

    return (
        <div className={styles.container}>
            <div className={styles.logo}>
                {text.split('').map((char, index) => (
                    <span
                        key={index}
                        className={styles.waveLetter}
                        style={{
                            animationDelay: `${index * 0.1}s`,
                            display: 'inline-block'
                        }}
                    >
                        {char}
                    </span>
                ))}
            </div>
            <Form />
        </div>
    );
};

export default Home;