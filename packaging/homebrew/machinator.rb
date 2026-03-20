class Machinator < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "https://github.com/jmcguigan10/Machinator"
  url "https://github.com/jmcguigan10/Machinator/releases/download/v0.7.0/machinator-0.7.0.tar.gz"
  sha256 "bfee9a9185917a5882e3916a81fd3cab7894d54dbcc17f3f841085bd6c52f059"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinator", shell_output("#{bin}/macht --help")
  end
end
